import glob
from src.pdf_loader import extract_text_from_pdfs, split_text
from src.embed_store import build_vectorstore

pdfs = glob.glob("data/*.pdf")
raw_text = extract_text_from_pdfs(pdfs)
chunks = split_text(raw_text)

build_vectorstore(chunks)
print("Chroma DB built")
