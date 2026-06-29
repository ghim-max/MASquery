import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Folder where the PDFs live
DOCS_DIR = Path(__file__).resolve().parent / "docs"
# Folder where the Chroma database will be saved locally
DB_DIR = Path(__file__).resolve().parent / "chroma_db"


def load_pdfs_from_docs() -> list:
    """Load all PDF files from the docs folder."""
    pdf_files = sorted(DOCS_DIR.rglob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {DOCS_DIR}.")
        return []

    documents = []

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        # Add the source file name to each page so we know where the text came from
        for page in pages:
            page.metadata["source"] = pdf_path.name

        documents.extend(pages)

    return documents


def build_vector_database() -> None:
    """Split documents into chunks, embed them, and store them in Chroma."""
    documents = load_pdfs_from_docs()

    if not documents:
        print("No documents were loaded. Nothing to index.")
        return

    # Split long documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} pages into {len(chunks)} chunks.")

    # Use a sentence-transformer model to create embeddings for each chunk
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Save the chunks and embeddings to a local Chroma database
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    print(f"Vector database created at: {DB_DIR}")


if __name__ == "__main__":
    build_vector_database()
