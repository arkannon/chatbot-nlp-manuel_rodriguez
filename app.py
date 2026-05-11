
import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.prompts import PromptTemplate

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / "gemini.env"
KB_PATH = BASE_DIR / "knowledge_base_nlp.txt"

load_dotenv(ENV_PATH)

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY was not found. Add it to gemini.env locally or to Streamlit Cloud Secrets.")
    st.stop()

st.set_page_config(
    page_title="Academic NLP Chatbot",
    page_icon="🤖"
)

st.title("Academic NLP Chatbot")

st.write(
    "This chatbot answers academic questions about NLP, LLMs, LangChain, "
    "ChromaDB, RAG and Streamlit using an external knowledge base."
)

with st.sidebar:
    st.subheader("Model configuration")
    selected_model = st.selectbox(
        "Generative model",
        ["gemini-3.1-flash-lite", "gemini-3-flash-preview"],
        index=0
    )
    st.caption("Use Gemini 3.1 Flash-Lite for normal testing because it has the highest available daily quota in your account.")

    st.subheader("Suggested questions")
    st.write("What is RAG?")
    st.write("What is ChromaDB?")
    st.write("Why is conversational memory important?")
    st.write("What metrics can be used to evaluate a chatbot?")
    st.warning("Avoid sending many questions quickly. Free tier models have RPM and daily limits.")

    clear_clicked = st.button("Clear conversation")

st.info(
    f"Current model: {selected_model}. Quota-safe settings: temperature=0.2, "
    "max_output_tokens=300 and retriever k=2."
)

if not KB_PATH.exists():
    st.error("The external knowledge base file knowledge_base_nlp.txt was not found.")
    st.stop()

with open(KB_PATH, "r", encoding="utf-8") as file:
    external_knowledge = file.read()


def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


documents = split_text(external_knowledge)

# Rebuild the chain if the selected model changes.
if st.session_state.get("selected_model") != selected_model:
    st.session_state.selected_model = selected_model
    st.session_state.pop("llm", None)
    st.session_state.pop("chatbot", None)
    st.session_state.pop("memory", None)

if "llm" not in st.session_state:
    st.session_state.llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.2,
        max_output_tokens=300,
        api_key=api_key
    )

if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

if "db" not in st.session_state:
    st.session_state.db = Chroma.from_texts(
        texts=documents,
        embedding=st.session_state.embeddings
    )

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an academic NLP chatbot.
Answer the question using only the retrieved context.
If the answer is not present in the context, say: "I do not have enough information in the knowledge base."
Use a clear and concise academic style.

Context:
{context}

Question:
{question}

Answer:
"""
)

if "chatbot" not in st.session_state:
    st.session_state.chatbot = ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=st.session_state.db.as_retriever(search_kwargs={"k": 2}),
        memory=st.session_state.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if clear_clicked:
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask a question about NLP:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    try:
        with st.spinner("Generating answer..."):
            response = st.session_state.chatbot.invoke({"question": user_input})
            answer = response["answer"]
    except Exception as error:
        answer = (
            "The model could not answer because the API quota or rate limit was reached. "
            "Please wait a few minutes, switch models in the sidebar, or try again tomorrow if the daily quota was exhausted."
        )
        st.error(str(error))

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)
