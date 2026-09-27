import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from core.document_loader import load_pdf
from core.text_chunker import split_documents
from core.vector_store import create_vector_store
from core.rag_pipeline import answer_question

load_dotenv()

st.set_page_config(
    page_title="DocMind",
    page_icon="📄",
    layout="wide",
)


def initialize_state():
    defaults = {
        "vector_store": None,
        "document_name": None,
        "document_pages": 0,
        "document_chunks": 0,
        "conversation": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def index_document(uploaded_file):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        documents = load_pdf(temp_path)
        chunks = split_documents(documents)
        vector_store = create_vector_store(chunks)

        st.session_state.vector_store = vector_store
        st.session_state.document_name = uploaded_file.name
        st.session_state.document_pages = len(documents)
        st.session_state.document_chunks = len(chunks)
        st.session_state.conversation = []

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def build_conversation_text():
    lines = [
        "DocMind Conversation",
        f"Document: {st.session_state.document_name}",
        "=" * 60,
        "",
    ]

    for message in st.session_state.conversation:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            lines.append("USER")
            lines.append("-" * 60)
            lines.append(content)
            lines.append("")

        elif role == "assistant":
            lines.append("DOCMIND")
            lines.append("-" * 60)
            lines.append(content)
            lines.append("")

            sources = message.get("sources", [])

            if sources:
                lines.append("RETRIEVED SOURCES")
                lines.append("-" * 60)

                for source in sources:
                    lines.append(
                        f"Page {source['page']} | Distance: {source['distance']:.4f}"
                    )

                lines.append("")

        lines.append("=" * 60)
        lines.append("")

    return "\n".join(lines)


initialize_state()


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.title("📄 DocMind")

st.caption("Ask questions about a PDF and get grounded answers with source pages.")


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:
        if st.session_state.document_name != uploaded_file.name:
            if st.button(
                "Index Document",
                use_container_width=True,
            ):
                with st.spinner("Reading and indexing document..."):
                    index_document(uploaded_file)

                st.success("Document indexed successfully.")

    if st.session_state.vector_store is not None:
        st.divider()

        st.subheader("Document Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Pages",
                st.session_state.document_pages,
            )

        with col2:
            st.metric(
                "Chunks",
                st.session_state.document_chunks,
            )

        st.caption(f"**Document:** {st.session_state.document_name}")

        st.divider()

        if st.button(
            "Clear Document",
            use_container_width=True,
        ):
            st.session_state.vector_store = None
            st.session_state.document_name = None
            st.session_state.document_pages = 0
            st.session_state.document_chunks = 0
            st.session_state.conversation = []
            st.rerun()


# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────

if st.session_state.vector_store is None:
    st.info(
        "Upload a PDF from the sidebar and click "
        "**Index Document** to start asking questions."
    )

    st.markdown(
        """
### How DocMind works

1. Upload a PDF
2. Extract text from every page
3. Split the document into searchable chunks
4. Create local embeddings
5. Store the chunks in FAISS
6. Retrieve the most relevant chunks
7. Generate a grounded answer with Gemma
8. Show the source pages used for the answer
"""
    )

else:
    st.success(f"Ready to answer questions about **{st.session_state.document_name}**.")

    # ─────────────────────────────────────────
    # Existing Conversation
    # ─────────────────────────────────────────

    for message in st.session_state.conversation:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])

        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(message["content"])

                sources = message.get("sources", [])

                if sources:
                    with st.expander("Retrieved Sources"):
                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):
                            st.markdown(f"**Source {index} — Page {source['page']}**")

                            st.caption(f"Distance: {source['distance']:.4f}")

                            st.write(source["content"])

                            if index < len(sources):
                                st.divider()

    # ─────────────────────────────────────────
    # New Question
    # ─────────────────────────────────────────

    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.conversation.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document and generating an answer..."):
                result = answer_question(
                    st.session_state.vector_store,
                    question,
                    k=5,
                )

            st.write(result["answer"])

            with st.expander("Retrieved Sources"):
                for index, source in enumerate(
                    result["sources"],
                    start=1,
                ):
                    st.markdown(f"**Source {index} — Page {source['page']}**")

                    st.caption(f"Distance: {source['distance']:.4f}")

                    st.write(source["content"])

                    if index < len(result["sources"]):
                        st.divider()

        st.session_state.conversation.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )

    # ─────────────────────────────────────────
    # Download Conversation
    # ─────────────────────────────────────────

    if st.session_state.conversation:
        st.divider()

        st.download_button(
            "Download Conversation",
            data=build_conversation_text(),
            file_name="docmind_conversation.txt",
            mime="text/plain",
        )
