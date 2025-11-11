"""
RAG (Retrieval-Augmented Generation) Pipeline
==============================================
Implements document retrieval and context injection for LLM queries.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
    from langchain_core.documents import Document
except ImportError:
    print("⚠️  LangChain not installed. Run: pip install langchain langchain-community langchain-openai langchain-text-splitters")
    raise


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for medical knowledge retrieval.
    
    This pipeline:
    1. Loads medical documents from local corpus
    2. Generates embeddings using OpenAI/other embedding models
    3. Stores embeddings in vector database (FAISS)
    4. Retrieves relevant context based on queries
    5. Injects context into LLM prompts
    """
    
    def __init__(
        self,
        corpus_path: str = "documents/sample_docs",
        embedding_model: str = "openai",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            corpus_path: Path to document corpus directory
            embedding_model: Embedding model to use ('openai', 'huggingface', etc.)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            top_k: Number of top documents to retrieve
        """
        self.corpus_path = corpus_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings(embedding_model)
        
        # Vector store (will be initialized when loading documents)
        self.vectorstore = None
        
        print(f"✓ RAG Pipeline initialized with {embedding_model} embeddings")
    
    def _initialize_embeddings(self, model: str):
        """
        Initialize embedding model.
        
        Args:
            model: Model type ('openai', 'huggingface', etc.)
            
        Returns:
            Embedding model instance
        """
        if model == "openai":
            # Requires OPENAI_API_KEY environment variable
            return OpenAIEmbeddings()
        elif model == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unsupported embedding model: {model}")
    
    def load_documents(self) -> List[Document]:
        """
        Load documents from corpus directory.
        Supports: .txt and .pdf files
        
        Returns:
            List of loaded documents
        """
        corpus_path = Path(self.corpus_path)
        
        if not corpus_path.exists():
            print(f"⚠️  Corpus path does not exist: {corpus_path}")
            print(f"Creating directory and using empty corpus")
            corpus_path.mkdir(parents=True, exist_ok=True)
            return []
        
        print(f"Loading documents from: {corpus_path}")
        
        all_documents = []
        
        # Load text files
        txt_files = list(corpus_path.glob("**/*.txt"))
        if txt_files:
            print(f"Loading {len(txt_files)} text files...")
            txt_loader = DirectoryLoader(
                str(corpus_path),
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            all_documents.extend(txt_loader.load())
        
        # Load PDF files
        pdf_files = list(corpus_path.glob("**/*.pdf"))
        if pdf_files:
            print(f"Loading {len(pdf_files)} PDF files...")
            for pdf_file in pdf_files:
                try:
                    pdf_loader = PyPDFLoader(str(pdf_file))
                    all_documents.extend(pdf_loader.load())
                    print(f"  ✓ Loaded: {pdf_file.name}")
                except Exception as e:
                    print(f"  ⚠️  Failed to load {pdf_file.name}: {e}")
        
        print(f"✓ Loaded {len(all_documents)} total documents ({len(txt_files)} txt, {len(pdf_files)} pdf)")
        
        return all_documents
    
    def build_vector_store(self, documents: Optional[List[Document]] = None):
        """
        Build vector store from documents.
        
        Args:
            documents: List of documents (if None, loads from corpus)
        """
        if documents is None:
            documents = self.load_documents()
        
        if not documents:
            print("⚠️  No documents to index. Vector store will be empty.")
            # Create empty vector store
            self.vectorstore = FAISS.from_texts(
                ["Placeholder document for empty corpus"],
                self.embeddings
            )
            return
        
        # Split documents into chunks
        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Create vector store
        print("Building vector store with embeddings...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("✓ Vector store built successfully")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve (uses default if None)
            
        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_vector_store() first.")
        
        k = top_k if top_k is not None else self.top_k
        
        # Retrieve similar documents
        docs = self.vectorstore.similarity_search(query, k=k)
        
        return docs
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[tuple]:
        """
        Retrieve relevant documents with similarity scores.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_vector_store() first.")
        
        k = top_k if top_k is not None else self.top_k
        
        # Retrieve with scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        return docs_with_scores
    
    def get_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Get formatted context string from retrieved documents.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        docs = self.retrieve(query, top_k)
        
        if not docs:
            return "No relevant medical documentation found."
        
        # Format context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"--- Reference {i} ---")
            context_parts.append(doc.page_content)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def save_vector_store(self, path: str):
        """
        Save vector store to disk.
        
        Args:
            path: Path to save vector store
        """
        if self.vectorstore is None:
            raise ValueError("No vector store to save")
        
        self.vectorstore.save_local(path)
        print(f"✓ Vector store saved to: {path}")
    
    def load_vector_store(self, path: str):
        """
        Load vector store from disk.
        
        Args:
            path: Path to saved vector store
        """
        self.vectorstore = FAISS.load_local(path, self.embeddings)
        print(f"✓ Vector store loaded from: {path}")

