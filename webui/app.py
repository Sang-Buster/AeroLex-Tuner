import asyncio
import logging

import ollama
import streamlit as st

# Configuration Variables
OLLAMA_HOST = "http://localhost:11434"  # Ollama server address
DEFAULT_MODEL_1 = "llama3.2:3b-instruct-q4_K_M"  # Default model 1
DEFAULT_MODEL_2 = "hf.co/Sang-Buster/atc-llama-gguf:Q4_K_M"  # Default model 2

# Set logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("watchdog").setLevel(logging.WARNING)  # Suppress watchdog warnings
logging.getLogger("httpcore").setLevel(logging.WARNING)  # Suppress httpcore warnings
logger = logging.getLogger(__name__)

# Page config
title = "🦙 Ollama Concurrent Model Comparison 🦙"
st.set_page_config(page_title=title, layout="wide")
st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)

# Initialize session state
if "messages1" not in st.session_state:
    st.session_state.messages1 = []
if "messages2" not in st.session_state:
    st.session_state.messages2 = []
if "input_disabled" not in st.session_state:
    st.session_state.input_disabled = False
if "model_1" not in st.session_state:
    st.session_state.model_1 = DEFAULT_MODEL_1
if "model_2" not in st.session_state:
    st.session_state.model_2 = DEFAULT_MODEL_2

# Initialize Ollama client
async_client = ollama.AsyncClient(host=OLLAMA_HOST)


# Helper functions
async def get_models():
    try:
        logger.debug("Attempting to connect to Ollama")
        response = await async_client.list()
        models = (
            [model.model for model in response.models]
            if hasattr(response, "models")
            else []
        )

        if not models:
            raise Exception("No models found")

        logger.debug(f"Found models: {models}")
        return models

    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {str(e)}")
        st.error(
            "Cannot connect to Ollama. Please make sure the port forwarding is active: `vega_ollama`"
        )
        st.stop()


def clear_everything():
    st.session_state.messages1 = []
    st.session_state.messages2 = []
    st.session_state.input_disabled = False


def update_model_1():
    st.session_state.model_1 = st.session_state.select_model_1


def update_model_2():
    st.session_state.model_2 = st.session_state.select_model_2


def process_prompt():
    prompt_text = st.session_state.chat_input
    if prompt_text.strip():
        st.session_state.messages1.append({"role": "user", "content": prompt_text})
        st.session_state.messages2.append({"role": "user", "content": prompt_text})
        st.session_state.input_disabled = True
        asyncio.run(main())


# Get available models
models = asyncio.run(get_models())

# Sidebar
with st.sidebar:
    model_1 = st.selectbox(
        "Model 1",
        options=models,
        key="select_model_1",
        index=models.index(st.session_state.model_1),
        on_change=update_model_1,
    )

    model_2 = st.selectbox(
        "Model 2",
        options=models,
        key="select_model_2",
        index=models.index(st.session_state.model_2),
        on_change=update_model_2,
    )

    st.markdown("<h5 class='prompt'>Prompt</h5>", unsafe_allow_html=True)
    prompt = st.chat_input(
        "Message Ollama",
        on_submit=process_prompt,
        key="chat_input",
        disabled=st.session_state.input_disabled,
    )

    st.button(
        "New Chat :speech_balloon:", on_click=clear_everything, use_container_width=True
    )


# Main chat interface
async def run_prompt(placeholder, model, message_history):
    client = ollama.AsyncClient(host=OLLAMA_HOST)

    try:
        with placeholder.container():
            for msg in message_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            assistant_msg = st.chat_message("assistant")
            response_placeholder = assistant_msg.empty()

            stream = await client.chat(
                model=model, messages=message_history, stream=True
            )

            full_response = ""
            async for chunk in stream:
                if chunk and "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    full_response += content
                    response_placeholder.markdown(full_response)

            message_history.append({"role": "assistant", "content": full_response})

    except Exception as e:
        logger.error(f"Error during chat: {str(e)}")
        st.error(f"Error communicating with model: {str(e)}")


async def main():
    try:
        tasks = [
            run_prompt(
                body_1,
                model=st.session_state.model_1,
                message_history=st.session_state.messages1,
            ),
            run_prompt(
                body_2,
                model=st.session_state.model_2,
                message_history=st.session_state.messages2,
            ),
        ]
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        st.session_state.input_disabled = False


# Display chat interface
col1, col2 = st.columns(2)
col1.write(f"# :blue[Model 1: {st.session_state.model_1}]")
col2.write(f"# :red[Model 2: {st.session_state.model_2}]")

body_1 = col1.container()
body_2 = col2.container()

with body_1:
    for message in st.session_state.messages1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with body_2:
    for message in st.session_state.messages2:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
