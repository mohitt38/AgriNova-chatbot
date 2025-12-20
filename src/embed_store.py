<<<<<<< HEAD
import os
import streamlit as st
=======
>>>>>>> 709e300 (updated)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_DB_DIR, EMBEDDEDING_MODEL_NAME

<<<<<<< HEAD
# Ensure DB directory exists
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDEDING_MODEL_NAME
    )

def is_db_built():
    """
    Reliable check: Chroma creates this file only after successful build
    """
    return os.path.exists(os.path.join(CHROMA_DB_DIR, "chroma.sqlite3"))

def build_vectorstore(chunks):
    embeddings = get_embeddings()
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
=======
# DO NOT use streamlit here

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDEDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
>>>>>>> 709e300 (updated)
    )

@st.cache_resource
def load_vectorstore():
<<<<<<< HEAD
    embeddings = get_embeddings()
=======
>>>>>>> 709e300 (updated)
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=get_embeddings()
    )
