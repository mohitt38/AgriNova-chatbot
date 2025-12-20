import streamlit as st
import unicodedata

from src.embed_store import load_vectorstore
from src.chatbot import ask_crop_expert


def clean_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


st.set_page_config(
    page_title="AgriNova – Crop Expert Chatbot",
    layout="wide"
)

st.title("🌱 AgriNova – Crop Expert Chatbot")
st.write("💬 Ask any crop-related question and get expert advice!")


@st.cache_resource(show_spinner=True)
def get_vectorstore():
    return load_vectorstore()


vectorstore = get_vectorstore()


# INIT CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []


# SHOW CHAT HISTORY (USE chat_message ONLY)
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])


# INPUT
user_input = st.chat_input("🌾 Ask in English / Hindi / Punjabi")

if user_input:
    user_input = clean_text(user_input)

    # SAVE & SHOW USER MESSAGE
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    st.chat_message("user").markdown(user_input)

    # STREAM BOT RESPONSE
    response_container = st.chat_message("assistant")
    full_response = ""

    for token in ask_crop_expert(
        user_input,
        vectorstore=vectorstore,
        stream=True
    ):
        text = token.content if hasattr(token, "content") else str(token)
        full_response += text
        response_container.markdown(full_response)

    # SAVE BOT RESPONSE
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
