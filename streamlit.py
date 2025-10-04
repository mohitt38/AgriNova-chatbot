import streamlit as st
import glob, os, unicodedata
from src.pdf_loader import extract_text_from_pdfs, split_text
from src.embed_store import build_vectorstore
from src.chatbot import ask_crop_expert

# ------------------------------
# Unicode normalization
# ------------------------------
def clean_text(text: str) -> str:
    """Normalize Hindi/Punjabi/English text to avoid hidden encoding issues"""
    return unicodedata.normalize("NFKC", text).strip()

# ------------------------------
# Page config
# ------------------------------
st.set_page_config("AGRIBOT - Crop Expert Chatbot", layout="wide")
st.title("🌱 AgriNova - Crop Expert Chatbot")
st.write("💬 Ask any crop-related question and get expert advice!")

# ------------------------------
# Build knowledge base once
# ------------------------------
if not os.path.exists("db/chroma_db"):
    pdfs = glob.glob("data/*.pdf")
    raw_text = extract_text_from_pdfs(pdfs)
    chunks = split_text(raw_text)
    build_vectorstore(chunks)
    st.success("Knowledge base built from PDFs ✅")

# ------------------------------
# Initialize chat history
# ------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.markdown("### How can I help you today? ")

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f" **User** {msg['content']}")
    else:
        st.markdown(f"AgriNova {msg['content']}")

# ------------------------------
# Input box
# ------------------------------
user_input = st.chat_input("🌾 INPUT THE QUERY (Type in ENGLISH/Hindi/Punjabi):")

if user_input:
    user_input = clean_text(user_input)

    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.markdown(f" **You:** {user_input}")

    with st.spinner("Thinking..."):
        response_placeholder = st.empty()
        full_response = ""

        # Streaming mode
        for token in ask_crop_expert(user_input, stream=True):
            # Extract text content if token is an object
            if hasattr(token, "content"):
                chunk_text = token.content
            elif isinstance(token, str):
                chunk_text = token
            else:
                chunk_text = str(token)  # fallback

            full_response += chunk_text
            response_placeholder.markdown(f"🤖 **AgriNova:** {full_response}")

        # Save final bot response after streaming ends
        st.session_state["messages"].append({"role": "bot", "content": full_response})
