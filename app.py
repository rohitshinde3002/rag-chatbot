
import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from src.loader import load_documents
from src.vectorstore import (
    create_vector_store,
    load_vector_store
)
from src.chatbot import get_qa_chain

# =========================
# LOAD ENV
# =========================

load_dotenv()

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enterprise RAG Chatbot")

# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):

        if not uploaded_files:
            st.warning(
                "Please upload at least one PDF."
            )
            st.stop()

        paths = []

        try:

            for file in uploaded_files:

                if file.size == 0:
                    st.error(
                        f"{file.name} is empty."
                    )
                    st.stop()

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp:

                    temp.write(
                        file.getvalue()
                    )

                    temp.flush()

                    paths.append(
                        temp.name
                    )

            with st.spinner(
                "Creating Knowledge Base..."
            ):

                chunks = load_documents(
                    paths
                )

                if len(chunks) == 0:

                    st.error(
                        "No text found in uploaded PDFs."
                    )

                    st.stop()

                create_vector_store(
                    chunks
                )

            st.success(
                "✅ Knowledge Base Created Successfully"
            )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )

# =========================
# CHAT HISTORY
# =========================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )

# =========================
# CHAT INPUT
# =========================

prompt = st.chat_input(
    "Ask something from uploaded documents..."
)

if prompt:

    # Check FAISS index exists

    if not os.path.exists(
        "faiss_index/index.faiss"
    ):

        st.error(
            "Please upload PDFs and click Process Documents first."
        )

        st.stop()

    # Show user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    try:

        # Load vector database

        db = load_vector_store()

        # Create QA chain

        chain = get_qa_chain(
            db
        )

        # Generate answer

        with st.spinner(
            "Thinking..."
        ):

            result = chain.invoke(
                {
                    "input": prompt
                }
            )

            answer = result["answer"]

        # Save assistant response

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # Show assistant response

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )

    except FileNotFoundError:

        st.error(
            "Please upload and process PDF documents first."
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )
