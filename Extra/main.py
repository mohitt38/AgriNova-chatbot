import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")

# ---------------------------
# PDF Text Extraction
# ---------------------------
def get_pdf_text(RAG):
    text = ""
    for pdf in RAG:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# ---------------------------
# Text Chunking
# ---------------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

# ---------------------------
# Chroma Vector Store with HuggingFace Embeddings
# ---------------------------
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create persistent ChromaDB
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vector_store.persist()
    return vector_store

# ---------------------------
# QA Chain with Gemini
# ---------------------------
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say:
    "Answer is not available in the context."
    Do not make up answers.

    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    return chain

# ---------------------------
# User Question → Retrieval + Answer
# ---------------------------
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Load persistent ChromaDB
    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    docs = vector_store.similarity_search(user_question, k=3)  # top 3 relevant chunks

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config("Chat PDF")
    st.header("📄 Chat with PDF using ChromaDB + Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed and stored in ChromaDB ✅")

if __name__ == "__main__":
    main()
