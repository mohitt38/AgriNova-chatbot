import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_DB_DIR, EMBEDDEDING_MODEL_NAME

os.makedirs(CHROMA_DB_DIR, exist_ok=True)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDEDING_MODEL_NAME
    )

def is_db_built():
    return os.path.exists(os.path.join(CHROMA_DB_DIR, "chroma.sqlite3"))

def build_vectorstore(chunks):
    embeddings = get_embeddings()
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    return vectorstore

@st.cache_resource
def load_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
