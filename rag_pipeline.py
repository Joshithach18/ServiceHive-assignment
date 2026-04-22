# rag_pipeline.py
# RAG pipeline — loads the AutoStream knowledge base, chunks it,
# embeds with a local sentence-transformer model, and stores in FAISS.

import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Default path relative to this file
_DEFAULT_KB = Path(__file__).parent / "knowledge_base" / "autostream_kb.md"


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for AutoStream knowledge base.

    Uses:
    - sentence-transformers/all-MiniLM-L6-v2  (local, ~80 MB, no API key needed)
    - FAISS in-memory vector store
    - Cosine similarity search

    Usage:
        rag = RAGPipeline()
        context = rag.retrieve("What is the Pro plan price?")
    """

    def __init__(self, kb_path: str = str(_DEFAULT_KB)):
        print("🔍  Initializing RAG pipeline (loading embeddings model)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._build_vectorstore(kb_path)
        print("✅  RAG pipeline ready.\n")

    # ------------------------------------------------------------------
    def _build_vectorstore(self, kb_path: str) -> None:
        """Load KB → split into chunks → embed → store in FAISS."""
        if not os.path.exists(kb_path):
            raise FileNotFoundError(
                f"Knowledge base file not found at: {kb_path}\n"
                "Make sure 'knowledge_base/autostream_kb.md' exists."
            )

        loader = TextLoader(kb_path, encoding="utf-8")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(documents)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print(f"   Indexed {len(chunks)} chunks from knowledge base.")

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 3) -> str:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Natural language query from the user
            k:     Number of chunks to return (default 3)

        Returns:
            Concatenated relevant text chunks separated by dividers
        """
        if not query.strip():
            return ""

        docs = self.vectorstore.similarity_search(query, k=k)
        if not docs:
            return ""

        return "\n\n---\n\n".join(doc.page_content for doc in docs)
