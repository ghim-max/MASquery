"""
RAG QA Assistant
A Streamlit app that answers questions about your PDF documents.

How it works:
  1. Load the Chroma vector database built by ingest.py
  2. Convert the user's question into a vector and find the closest document chunks
  3. Send the question + those chunks to Claude as context
  4. Display Claude's answer and which documents it came from
"""

import os
from pathlib import Path

import anthropic          # Official Anthropic Python SDK
import streamlit as st
from dotenv import load_dotenv                         # reads the .env file
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # fixed: was langchain_community

# ── Load environment variables ────────────────────────────────────────────────
# This reads the ANTHROPIC_API_KEY (and anything else) from your .env file and
# puts it into the process environment so os.getenv() can find it later.
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
# Where ingest.py stored the Chroma database on disk
DB_DIR = Path(__file__).resolve().parent / "chroma_db"

# Must be the same model name used in ingest.py — the vector spaces must match
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many document chunks to retrieve for each question
TOP_K = 4


# ── Step 1: load the vector database (once) ───────────────────────────────────
@st.cache_resource
def get_vector_store():
    """
    Open the Chroma database and return it.

    @st.cache_resource runs this function exactly once for the lifetime of the
    Streamlit server process.  On every subsequent user interaction, Streamlit
    returns the already-loaded database instead of re-reading it from disk,
    which is much faster.
    """
    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"No vector database found at '{DB_DIR}'. "
            "Please run  python3 ingest.py  first to build the database."
        )

    # HuggingFaceEmbeddings converts text to numerical vectors.
    # We need the exact same model here as in ingest.py so the numbers line up.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Open the existing database — do NOT call Chroma.from_documents() here
    # because that would re-build it.  We just want to read what's already there.
    return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)


# ── Step 2 + 3: ask Claude a question with context ────────────────────────────
def ask_claude(question: str, context_docs: list) -> str:
    """
    Send the user's question and the retrieved document chunks to Claude.
    Returns Claude's answer as a plain string.

    We use streaming so the HTTP request doesn't time out for long answers.
    get_final_message() waits for the full stream and returns the complete text.
    """
    # Build a single block of context text from all retrieved chunks.
    # Each chunk is separated by a horizontal rule so Claude can tell them apart.
    context_text = "\n\n---\n\n".join(chunk.page_content for chunk in context_docs)

    # The system prompt tells Claude its role and how to use the context.
    system_prompt = (
        "You are a helpful assistant. "
        "Answer the user's question using ONLY the information in the provided context. "
        "If the context does not contain enough information, say so clearly. "
        "Do not make up facts."
    )

    # The user message packages the context and the question together.
    user_message = (
        f"Context (excerpts from relevant documents):\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )

    # Create an Anthropic client.
    # If ANTHROPIC_API_KEY is in the environment (loaded from .env above),
    # the client picks it up automatically — no need to pass it explicitly.
    client = anthropic.Anthropic()

    # stream() sends the request and yields text as Claude produces it.
    # get_final_message() collects everything and returns the complete response.
    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        final = stream.get_final_message()

    # The response content is a list of blocks.  We want the text blocks.
    answer_parts = []
    for block in final.content:
        if hasattr(block, "text"):
            answer_parts.append(block.text)

    return "".join(answer_parts)


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="MASQuery", page_icon="📄")
st.title("MASQuery")
st.write(
    "Ask questions about MAS fintech regulations. "
    "Powered by retrieval-augmented generation (RAG) to ground answers in the source documents."
)

# Text input box for the user's question
question = st.text_input(
    "Your question",
    placeholder="e.g. What are the main topics covered in the documents?",
)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question before clicking Ask.")
    else:
        try:
            # ── Step 1: load the database ──────────────────────────────────
            vector_store = get_vector_store()

            # ── Step 2: find the most relevant chunks ──────────────────────
            # similarity_search converts the question to a vector, then finds
            # the k database chunks whose vectors are closest to it.
            with st.spinner("Searching for relevant passages…"):
                relevant_docs = vector_store.similarity_search(question, k=TOP_K)

            if not relevant_docs:
                st.info(
                    "No relevant passages were found in the database. "
                    "Try rephrasing your question."
                )
                st.stop()

            # ── Step 3: show which source files were used ──────────────────
            # Each chunk carries metadata["source"] set by ingest.py.
            source_names = sorted(
                {doc.metadata.get("source", "unknown") for doc in relevant_docs}
            )
            st.caption(f"Sources consulted: {', '.join(source_names)}")

            # ── Step 4: ask Claude and display the answer ──────────────────
            with st.spinner("Asking Claude…"):
                answer = ask_claude(question, relevant_docs)

            st.subheader("Answer")
            st.write(answer)

            # ── Step 5: show the raw passages (optional, collapsed by default)
            with st.expander("View retrieved passages"):
                for i, doc in enumerate(relevant_docs, start=1):
                    source = doc.metadata.get("source", "unknown")
                    st.markdown(f"**Passage {i}** — *{source}*")
                    st.write(doc.page_content)
                    st.divider()

        except FileNotFoundError as exc:
            st.error(str(exc))
        except anthropic.AuthenticationError:
            st.error(
                "Invalid or missing API key. "
                "Make sure ANTHROPIC_API_KEY is set correctly in your `.env` file."
            )
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
