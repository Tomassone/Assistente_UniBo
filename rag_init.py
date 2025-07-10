from llama_index.core import VectorStoreIndex, StorageContext, \
    load_index_from_storage, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain.memory import ConversationBufferMemory
import os

DOC_DIR = "./data"
VECTOR_STORE_DIR = "./vector_store"
LLM_MODEL = "ifioravanti/llamantino-2:7b-chat-ultrachat-it-q4_0"

# RAG COMPONENTS INITIALIZATION
def initialize_system():
    llm = Ollama(model=LLM_MODEL, request_timeout=300.0)
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )

    memory = ConversationBufferMemory(
        return_messages=True,
        max_token_limit=1000,
        memory_key="chat_history"
    )

    # BUILD OR LOAD INDEX
    if os.path.exists(VECTOR_STORE_DIR):
        print("Carico l'indice generato in precedenza...")
        storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORE_DIR)
        index = load_index_from_storage(storage_context, llm=llm, embed_model=embed_model)
    else:
        print("Genero un nuovo indice a partire dalle risorse date...")
        documents = SimpleDirectoryReader(DOC_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)
        index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
        print("Indice generato e salvato!")

    # QUERY LOOP
    query_engine = index.as_query_engine(llm=llm)
    return query_engine, memory
