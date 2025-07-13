# Assistente_UniBo
[Repository ufficiale del progetto "Assistente UniBo"]

Il chatbot “Assistente UniBo” utilizza le tecnologie RAG e LLM allo scopo di rispondere interattivamente alle domande degli utenti sul regolamento ufficiale dell’Università di Bologna. 
Nello specifico, il chatbot sfrutta: 
- un LLM della famiglia LLaMAntino per la creazione del testo; 
- le librerie open source LangChain e LlamaIndex per la creazione della pipeline RAG (composta da caricamento dei documenti, indicizzazione e ricerca semantica degli stessi, seguita infine della generazione delle risposte); 
- la libreria Streamlit per la realizzazione del front-end web dell'applicativo; 
- la piattaforma Ollama per il deployment locale del modello.
