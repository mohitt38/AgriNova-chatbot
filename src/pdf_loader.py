from pypdf import PdfReader
from langchain.text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdfs(pdf_paths):
    text = ""
    for path in pdf_paths:
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
