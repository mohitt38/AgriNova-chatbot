import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_DIR = "db/chroma_db"
EMBEDDEDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODEL_NAME = "gemini-2.5-pro"
TEMPERATURE = 0.3