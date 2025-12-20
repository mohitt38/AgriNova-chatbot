import streamlit as st
import glob, os, unicodedata

from src.pdf_loader import extract_text_from_pdfs, split_text
from src.embed_store import build_vectorstore, load_vectorstore, is_db_built
from src.chatbot import ask_crop_expert

# ------------------------------
# Unicode normalization
# ------------------------------
def clean_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()

# ------------------------------
# Page config
# ------------------------------
st.set_page_config("AGRINOVA - Crop Expert Chatbot", layout="wide")
st.title("🌱 AgriNova - Crop Expert Chatbot")
st.write("💬 Ask any crop-related question and get expert advice!")

# ------------------------------
# Build knowledge base ONLY ONCE
# ------------------------------
if not is_db_built():
    with st.spinner("Building knowledge base from PDFs (one-time)..."):
        pdfs = glob.glob("data/*.pdf")
        raw_text = extract_text_from_pdfs(pdfs)
        chunks = split_text(raw_text)
        build_vectorstore(chunks)
        st.success("Knowledge base built successfully ✅")

# ------------------------------
# Load vectorstore (cached)
# ------------------------------
vectorstore = load_vectorstore()

# ------------------------------
# Initialize chat history
# ------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.markdown("### How can I help you today?")

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **AgriNova:** {msg['content']}")

# ------------------------------
# Chat input
# ------------------------------
user_input = st.chat_input("🌾 Ask in English / Hindi / Punjabi")

if user_input:
    user_input = clean_text(user_input)

    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )

    with st.spinner("Thinking..."):
        response_placeholder = st.empty()
        full_response = ""

        for token in ask_crop_expert(
            user_input,
            vectorstore=vectorstore,
            stream=True
        ):
            if hasattr(token, "content"):
                text = token.content
            elif isinstance(token, str):
                text = token
            else:
                text = str(token)

            full_response += text
            response_placeholder.markdown(
                f"🤖 **AgriNova:** {full_response}"
            )

        st.session_state["messages"].append(
            {"role": "bot", "content": full_response}
        )
