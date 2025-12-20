import streamlit as st
import unicodedata

from src.embed_store import load_vectorstore
from src.chatbot import ask_crop_expert

# ------------------------------
# Unicode normalization
# ------------------------------
def clean_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()

# ------------------------------
# Page config
# ------------------------------
st.set_page_config("AGRIBOT - Crop Expert Chatbot", layout="wide")
st.title("🌱 AgriNova - Crop Expert Chatbot")
st.write("💬 Ask any crop-related question and get expert advice!")

# ------------------------------
# LOAD VECTORSTORE (READ-ONLY)
# ------------------------------
# ⚠️ NO BUILD LOGIC HERE
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
# Input box
# ------------------------------
user_input = st.chat_input(
    "🌾 INPUT THE QUERY (English / Hindi / Punjabi)"
)

if user_input:
    user_input = clean_text(user_input)

    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )

    response_placeholder = st.empty()
    full_response = ""

    with st.spinner("Thinking..."):
        for token in ask_crop_expert(
            user_input,
            vectorstore=vectorstore,
            stream=True
        ):
            text = token.content if hasattr(token, "content") else str(token)
            full_response += text
            response_placeholder.markdown(
                f"🤖 **AgriNova:** {full_response}"
            )

    st.session_state["messages"].append(
        {"role": "bot", "content": full_response}
    )
