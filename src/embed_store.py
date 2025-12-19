import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDEDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
from src.config import CHROMA_DB_DIR, EMBEDDEDING_MODEL_NAME

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDEDING_MODEL_NAME)
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDEDING_MODEL_NAME)
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
