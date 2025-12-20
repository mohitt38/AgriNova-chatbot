import os
import glob
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

from src.pdf_loader import extract_text_from_pdfs, split_text
from src.config import CHROMA_DB_DIR, EMBEDDEDING_MODEL_NAME
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDEDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore():
    if os.path.exists(CHROMA_DB_DIR):
        return Chroma(
            collection_name="agrinova_collection",
            persist_directory=CHROMA_DB_DIR,
            embedding_function=get_embeddings(),
            client_settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True,
            ),
        )
    pdfs = glob.glob("data/*.pdf")
    if not pdfs:
        raise RuntimeError("No PDFs found in data/ folder")

    raw_text = extract_text_from_pdfs(pdfs)
    chunks = split_text(raw_text)

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=get_embeddings(),
        collection_name="agrinova_collection",
        persist_directory=CHROMA_DB_DIR,
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True,
        ),
    )

    vectorstore.persist()
    return vectorstore
