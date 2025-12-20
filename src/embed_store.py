from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

from src.config import CHROMA_DB_DIR, EMBEDDEDING_MODEL_NAME


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDEDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore():
    return Chroma(
        collection_name="agrinova_collection",
        persist_directory=CHROMA_DB_DIR,
        embedding_function=get_embeddings(),
        client_settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True,
        ),
    )
