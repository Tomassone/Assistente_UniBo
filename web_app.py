import streamlit as st
from rag_init import initialize_system

# BACKEND INITIALIZATION (from rag_init.py)
query_engine, memory = initialize_system()

st.set_page_config(page_title="Assistente UniBo", page_icon="🎓", layout="centered", initial_sidebar_state="expanded")
st.title("🎓 Assistente UniBo")
st.caption("Risponde a domande sul regolamento dell'Ateneo!")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/d/d0/Seal_of_the_University_of_Bologna.svg", width=200)
st.sidebar.title("Alma Mater Studiorum - Università di Bologna")

# SESSION STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ciao! Come posso aiutarti?"}]
    st.session_state.memory = memory

# PAST CONVERSATION HANDLING
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# CHAT INPUT HANDLING:
if prompt := st.chat_input("Scrivi la tua domanda..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    # QUERY HANDLING
    with st.spinner("Analizzando la tua domanda..."):
        try:
            # PREPARE QUERY WITH CONTEXT
            history = st.session_state.memory.load_memory_variables({})["chat_history"]
            context = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
            augmented_input = f"Contesto della conversazione:\n{context}\n\nNuova domanda: {prompt}"

            response = query_engine.query(augmented_input)
            response_str = str(response)

            st.session_state.messages.append({"role": "assistant", "content": response_str})
            st.session_state.memory.chat_memory.add_user_message(prompt)
            st.session_state.memory.chat_memory.add_ai_message(response_str)

            # FORCE UI UPDATE
            st.rerun()

        except Exception as e:
            st.error(f"Errore del sistema: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Mi dispiace, c'è stato un errore."})
