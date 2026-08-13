import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIRECTORY = "db/chroma_db"


# ---------- Ressources mises en cache (chargées une seule fois) ----------

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_resource
def load_vectorstore(_embedding_model):
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=_embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


@st.cache_resource
def load_llm():
    return ChatOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        model="openai/gpt-oss-20b",
    )


def get_retriever(_vectorstore, k=5, score_threshold=0.3):
    return _vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )


def answer_question(query, retriever, llm):
    relevant_docs = retriever.invoke(query)

    if not relevant_docs:
        return "Je n'ai pas trouvé d'information pertinente dans les documents pour répondre à cette question.", []

    combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    result = llm.invoke(messages)
    return result.content, relevant_docs


# ---------- Interface Streamlit ----------

st.set_page_config(page_title="RAG Assistant", page_icon="🔎", layout="centered")
st.title("🔎 Assistant RAG")
st.caption("Posez une question, la réponse est générée à partir de vos documents.")

# Vérifie que la base vectorielle existe
if not os.path.exists(PERSIST_DIRECTORY):
    st.error(
        f"Le dossier '{PERSIST_DIRECTORY}' est introuvable. "
        f"Lancez d'abord `python pipline_ingestion.py` pour créer la base vectorielle."
    )
    st.stop()

# Chargement des ressources (caché après le premier appel)
with st.spinner("Chargement du modèle d'embeddings et de la base vectorielle..."):
    embedding_model = load_embedding_model()
    vectorstore = load_vectorstore(embedding_model)
    llm = load_llm()

retriever = get_retriever(vectorstore)

# Historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affiche l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources utilisées"):
                for i, doc in enumerate(msg["sources"], 1):
                    st.markdown(f"**Document {i}** — `{doc.metadata.get('source', 'inconnu')}`")
                    st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

# Zone de saisie
if query := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération de la réponse..."):
            try:
                answer, sources = answer_question(query, retriever, llm)
            except Exception as e:
                answer = f"⚠️ Une erreur est survenue : {e}"
                sources = []

            st.markdown(answer)
            if sources:
                with st.expander("📄 Sources utilisées"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Document {i}** — `{doc.metadata.get('source', 'inconnu')}`")
                        st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

# Bouton pour réinitialiser la conversation
if st.session_state.messages:
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()