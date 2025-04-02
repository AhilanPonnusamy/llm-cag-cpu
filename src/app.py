import html
import os

import pymupdf4llm
import streamlit as st
from openai import OpenAI

# Constant for default values
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1
MODEL = "google/gemma-3-1b-it"

st.title("LLM Chatbot CAG Assistant")


@st.cache_resource
def init_client() -> OpenAI:
    """Initialize the OpenAI client"""
    return OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )


client = init_client()


def save_file(uploaded_file) -> str:
    """Save uploaded file to the 'files' directory."""

    files_dir = os.path.join(os.path.dirname(__file__), "..", "files")
    os.makedirs(files_dir, exist_ok=True)  # Create the directory if it doesn't exist

    file_path = os.path.join(files_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_uploaded_file(uploaded_file) -> str | None:
    """Process the uploaded file and extract context."""
    try:
        file_path = save_file(uploaded_file)
        markdown = pymupdf4llm.to_markdown(file_path)
        return html.escape(markdown)  # Escape HTML to prevent injection
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None


def build_prompt(question: str, context: str | None) -> str:
    """Construct the prompt for the chatbot."""
    if context:
        return (
            "Answer the following question using the context provided from an internal, "
            "confidential document. Do not repeat anything from the context. "
            "Do not provide harmful or illegal advice.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}"
        )
    else:
        return f"Answer the following question: {question}"


def get_chat_messages(prompt: str, context: str) -> list[dict]:
    """Prepare the chat messages for the API call."""
    recent_messages = st.session_state.messages[-3:-1]
    system_message = {
        "role": "system",
        "content": "You are a helpful and friendly assistant.",
    }
    user_message = {"role": "user", "content": build_prompt(prompt, context)}
    return [system_message] + recent_messages + [user_message]


def get_response(messages: list[dict], temperature: float, max_tokens: int) -> str:
    """Call the OpenAI API to get a response."""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error fetching response: {e}")
        return "I'm sorry, I couldn't process your request."


with st.sidebar:
    max_completion_tokens = st.number_input(
        "max_completion_tokens", 128, 8192, DEFAULT_MAX_TOKENS
    )
    temperature = st.number_input("temperature", 0.0, 2.0, DEFAULT_TEMPERATURE)
    uploaded_file = st.file_uploader(
        "Upload file for context",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=False,
    )

    doc_context = process_uploaded_file(uploaded_file) if uploaded_file else None


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything!"):
    prompt = prompt.strip()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        messages = get_chat_messages(prompt, doc_context)
        answer = get_response(messages, temperature, max_completion_tokens)
        st.markdown(answer)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
