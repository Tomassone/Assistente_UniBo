import os
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pandas as pd
import re
from typing import List, Dict

DOC_DIR = "./data"
LLM_MODEL = "ifioravanti/llamantino-2:7b-chat-ultrachat-it-q4_0"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

llm = Ollama(
    model=LLM_MODEL,
    base_url="http://localhost:11434",
    timeout=300,
    temperature=0.3,
    num_ctx=2048
)

embed_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def load_documents():
    loader = DirectoryLoader(
        DOC_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return text_splitter.split_documents(docs)


# TEST Q/A GENERATION (+ CALL TO FALLBACK FUNCTION FOR ERROR HANDLING)    
def generate_test_set(split_docs, num_questions=10):
    try:
        sample_docs = split_docs[:3]
        context_text = "\n\n".join([doc.page_content[:500] for doc in sample_docs])
        
        # FORMAT USED TO GENERATE TEST Q/A
        prompt_template = """
        DEVI generare ESATTAMENTE {num_questions} coppie di domande e risposte.
        USA SOLO QUESTO FORMATO SENZA ECCEZIONI:

        INIZIO ESEMPIO
        1. Q: [Testo della domanda specifica sul contesto]
        1. A: [Testo della risposta precisa dal contesto]
        2. Q: [Seconda domanda specifica]
        2. A: [Seconda risposta precisa]
        FINE ESEMPIO

        CONTESTO:
        {context}

        REGOLE:
        - Numerare ogni coppia (1., 2., ...)
        - Usare "Q:" e "A:" per ogni domanda/risposta
        - Le domande devono essere SPECIFICHE al contesto
        - NESSUN altro testo oltre alle domande/risposte numerate
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({
            "context": context_text,
            "num_questions": num_questions
        })

        if not all(x in result for x in ["Q:", "A:", "1.", "2."]):
            raise ValueError("LLM deviated from required format")
            
        return parse_qa(result)
        
    except Exception as e:
        print(f"Question generation failed: {str(e)}")
        print("Using fallback questions...")
        return create_fallback_questions(sample_docs)

def parse_qa(text):
    qa_pairs = []
    current_pair = {"question": "", "answer": ""}
    
    # NORMALIZE TEXT FORMAT
    text = text.replace("?", "? ")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    # SUPPORTED PATTERNS
    number_pattern = r"^\d+\.?\s*"
    q_patterns = [r"Q[:.]", r"Domanda[:.]", r"\?$"]
    a_patterns = [r"A[:.]", r"Risposta[:.]"]
    
    for line in lines:
        try:
            # REMOVE NUMBERING
            clean_line = re.sub(number_pattern, "", line)
            
            # CHECK PRESENCE OF QUESTIONS
            if any(re.search(pattern, clean_line, re.IGNORECASE) for pattern in q_patterns):
                if current_pair["question"] and current_pair["answer"]:
                    qa_pairs.append(current_pair)
                    current_pair = {"question": "", "answer": ""}
                
                # EXTRACT QUESTION TEXT
                question_text = clean_line.split(":", 1)[1] if ":" in clean_line else clean_line
                current_pair["question"] = question_text.strip()
            
            # CHECK PRESENCE OF ANSWERS
            elif any(re.search(pattern, clean_line, re.IGNORECASE) for pattern in a_patterns):
                # EXTRACT ANSWER TEXT
                answer_text = clean_line.split(":", 1)[1] if ":" in clean_line else clean_line
                current_pair["answer"] = answer_text.strip()
                
        except Exception as e:
            print(f"Warning: Error parsing line - {str(e)}")
            continue
    
    if current_pair["question"] and current_pair["answer"]:
        qa_pairs.append(current_pair)
    
    if not qa_pairs:
        print(f"No Q&A pairs found in text:\n{text[:200]}...")
    else:
        print(f"Successfully parsed {len(qa_pairs)} Q&A pairs")
    
    return qa_pairs

def create_fallback_questions(docs):
    questions = []
    
    for i, doc in enumerate(docs[:5]):  # Limit to 5 docomunets to get 10 questions
        content = doc.page_content[:500]  # First 500 chars
        
        questions.extend([
            {
                "question": f"Qual è il contenuto principale del documento {i+1}?",
                "answer": f"Il documento contiene informazioni su: {content[:150]}..."
            },
            {
                "question": f"Quali sono i dettagli specifici discussi nel documento {i+1}?",
                "answer": f"Nel documento si discute dettagliatamente di: {content[100:250]}..."
            }
        ])
    
    print(f"Created {len(questions)} fallback questions")
    return questions

class LocalEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, prediction, reference, question):
        """Evaluate prediction against reference for given question."""
        prompt_text = f"""
        Valuta la qualità di questa risposta da 0 a 100:
        
        Domanda: {question}
        Risposta di riferimento: {reference}
        Risposta da valutare: {prediction}
        
        Fornisci solo un punteggio numerico tra 0 e 100.
        """

        try:
            response = self.llm.invoke(prompt_text)
            # NUMERIC SCORE EXTRACTION
            import re
            numbers = re.findall(r'\d+', str(response))
            if numbers:
                score = int(numbers[0])
                return {"score": score / 100}
            else:
                return {"score": 0}
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            return {"score": 0}

def evaluate_rag(test_set, vector_store):
    if not test_set:
        print("Error: No test questions available for evaluation")
        return pd.DataFrame()
    
    evaluator = LocalEvaluator(llm)
    results = []

    print(f"Evaluating {len(test_set)} questions...")
    
    for i, qa in enumerate(test_set):
        print(f"Processing question {i+1}/{len(test_set)}")
        
        try:
            # RELEVANT DOCS RETREIVAL
            docs = vector_store.similarity_search(qa["question"], k=4)
            context = "\n\n".join([d.page_content for d in docs])

            prompt_text = f"""
            Rispondi a questa domanda basandoti solo sul contesto dato:
            Domanda: {qa["question"]}

            Contesto:
            {context}

            Rispondi brevemente:
            """

            prediction = llm.invoke(prompt_text)

            # PREDICTION EVALUATION
            score = evaluator.evaluate(
                prediction=prediction,
                reference=qa["answer"],
                question=qa["question"]
            )

            results.append({
                "question": qa["question"],
                "score": score["score"],
                "prediction": prediction,
                "reference": qa["answer"],
                "sources": [d.metadata.get("source", "unknown") for d in docs]
            })
            
        except Exception as e:
            print(f"Error processing question {i+1}: {str(e)}")
            results.append({
                "question": qa["question"],
                "score": 0.0,
                "prediction": f"Error: {str(e)}",
                "reference": qa["answer"],
                "sources": []
            })

    return pd.DataFrame(results)

def visualize_performance(results_df):
    try:
        # CONVERT TO DATAFRAME, IF NECESSARY (ERROR HANDLING)
        if not isinstance(results_df, pd.DataFrame):
            results_df = pd.DataFrame(results_df)

        # CHECK FOR REQUIRED COLUMNS (ERROR HANDLING)
        if 'question' not in results_df.columns or 'score' not in results_df.columns:
            print("Error: Missing required columns (question, score)")
            return

        # CLEAN AND PREPARE DATA (ERROR HANDLING)
        results_df = results_df.copy()
        results_df['score'] = pd.to_numeric(results_df['score'], errors='coerce')
        results_df = results_df.dropna(subset=['score'])
        
        # BUILD DISPLAY TABLE
        display_df = pd.DataFrame({
            'Domanda #': [f"Q{i+1}" for i in range(len(results_df))],
            'Domanda': results_df['question'].str[:80] + '...',  # Truncate long questions
            'Punteggio': results_df['score'].round(3),
            'Giudizio': ['✓ Buono' if score >= 0.7 else '⚠ Medio' if score >= 0.4 else '✗ Scarso' 
                      for score in results_df['score']]
        })
        
        # PRINT THE DISPLAY TABLE
        print("\n" + "="*100)
        print("RISULTATI DI VALUTAZIONE PERFORMANCE DEL SISTEMA RAG")
        print("="*100)
        print(display_df.to_string(index=False, max_colwidth=80))
        
        # PRINT SUMMARY STATISTICS
        avg_score = results_df['score'].mean()
        print("\n" + "-"*50)
        print("STATISTICHE RIASSUNTIVE: ")
        print("-"*50)
        print(f"Numero domande: {len(results_df)}")
        print(f"Punteggio medio: {avg_score:.3f}")
        print(f"Punteggio massimo: {results_df['score'].max():.3f}")
        print(f"Punteggio peggiore: {results_df['score'].min():.3f}")
        print("="*100)
        
    except Exception as e:
        print(f"Error displaying results: {str(e)}")
        print("\nRaw data preview:")
        print(results_df.head() if isinstance(results_df, pd.DataFrame) else str(results_df)[:500])

def main():
    try:
        if not os.path.exists(DOC_DIR):
            print(f"Error: Directory {DOC_DIR} not found. Please create it and add PDF files.")
            return
        
        print("Caricando e processando i documenti...")
        split_docs = load_documents()
        
        if not split_docs:
            print("Error: No documents found. Please add PDF files to the data directory.")
            return
            
        print(f"Caricati {len(split_docs)} chunk di documenti")

        print("Creando il vector store...")
        vector_store = FAISS.from_documents(split_docs, embed_model)
        
        print("Generando le domande di test...")
        test_set = generate_test_set(split_docs)
        
        if not test_set:
            print("Error: Could not generate any test questions. Check your LLM connection.")
            return
            
        print(f"Sono state generate {len(test_set)} domande di test")
        
        print("\Domande di test:")
        for i, qa in enumerate(test_set[:3]):
            print(f"Q{i+1}: {qa['question']}")
            print(f"A{i+1}: {qa['answer'][:100]}...")
            print()

        print("Valutando il sistema RAG...")
        results = evaluate_rag(test_set, vector_store)
        
        if results.empty:
            print("Error: No evaluation results generated.")
            return
        
        results.to_csv("performance.csv", index=False)
        print("Risultati salvati in performance.csv")

        print("Visualizzando i risultati...")
        visualize_performance(results)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
