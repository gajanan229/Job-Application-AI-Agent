"""RAG (Retrieval-Augmented Generation) processor for the Application Factory."""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tempfile

# PDF processing
import PyPDF2
import pdfplumber

# Text processing and embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

# Vector store
import faiss
import numpy as np

# Configuration and logging
from config.settings import config
from config.logging_config import get_logger, log_processing_step
from utils.error_handlers import (
    RAGError, validate_file_operation, validate_api_operation, 
    error_handler
)

logger = get_logger(__name__)


class PDFProcessor:
    """Handles PDF text extraction using multiple methods for robustness."""
    
    @staticmethod
    @validate_file_operation
    @log_processing_step("PDF text extraction")
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from PDF using multiple extraction methods.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
            
        Raises:
            RAGError: If text extraction fails
        """
        try:
            # Try pdfplumber first (better for complex layouts)
            text = PDFProcessor._extract_with_pdfplumber(pdf_path)
            
            if not text.strip():
                logger.warning("pdfplumber extraction yielded no text, trying PyPDF2")
                text = PDFProcessor._extract_with_pypdf2(pdf_path)
            
            if not text.strip():
                raise RAGError(
                    "Failed to extract text from PDF",
                    details=f"Both pdfplumber and PyPDF2 failed for {pdf_path}",
                    user_message="Could not extract text from the PDF. The file may be corrupted or contain only images."
                )
            
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            if isinstance(e, RAGError):
                raise
            raise RAGError(
                f"PDF text extraction failed: {e}",
                details=str(e),
                user_message="Failed to extract text from PDF. Please ensure the file is a valid PDF document."
            )
    
    @staticmethod
    def _extract_with_pdfplumber(pdf_path: str) -> str:
        """Extract text using pdfplumber (better for tables and complex layouts)."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            logger.debug(f"pdfplumber extraction failed: {e}")
            
        return text.strip()
    
    @staticmethod
    def _extract_with_pypdf2(pdf_path: str) -> str:
        """Extract text using PyPDF2 (fallback method)."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            logger.debug(f"PyPDF2 extraction failed: {e}")
            
        return text.strip()
    
    @staticmethod
    def analyze_pdf_content(text: str) -> Dict[str, Any]:
        """
        Analyze extracted PDF content to provide insights.
        
        Args:
            text: Extracted text from PDF
            
        Returns:
            Dictionary with content analysis
        """
        lines = text.split('\n')
        words = text.split()
        
        analysis = {
            'total_characters': len(text),
            'total_words': len(words),
            'total_lines': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'average_words_per_line': len(words) / max(len(lines), 1),
            'has_email': '@' in text,
            'has_phone': any(char.isdigit() for char in text),
            'has_bullets': any(marker in text for marker in ['•', '◦', '-', '*']),
            'estimated_pages': max(1, len(text) // 2000)  # Rough estimate
        }
        
        return analysis


class TextChunker:
    """Handles text chunking for RAG processing."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    @log_processing_step("Text chunking")
    def chunk_text(self, text: str, source_type: str = "document") -> List[Document]:
        """
        Split text into chunks for RAG processing.
        
        Args:
            text: Text to be chunked
            source_type: Type of source document (resume, job_description, etc.)
            
        Returns:
            List of Document objects with metadata
        """
        try:
            # Create documents with metadata
            documents = []
            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source_type": source_type,
                            "chunk_index": i,
                            "chunk_size": len(chunk),
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Created {len(documents)} chunks from {source_type}")
            return documents
            
        except Exception as e:
            raise RAGError(
                f"Text chunking failed: {e}",
                details=str(e),
                user_message="Failed to process text for analysis. Please try again."
            )


class EmbeddingManager:
    """Manages embeddings creation and vector operations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize embedding manager.
        
        Args:
            api_key: Google API key for embeddings
        """
        self.api_key = api_key or config.google_api_key
        self.embeddings = None
        self._initialize_embeddings()
    
    @validate_api_operation
    def _initialize_embeddings(self):
        """Initialize Google Generative AI embeddings."""
        try:
            if not self.api_key:
                raise RAGError(
                    "Google API key not configured",
                    user_message="Google API key is required for embeddings. Please configure your API key."
                )
            
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=config.embedding_model,
                google_api_key=self.api_key
            )
            
            logger.info(f"Initialized embeddings with model: {config.embedding_model}")
            
        except Exception as e:
            if isinstance(e, RAGError):
                raise
            raise RAGError(
                f"Failed to initialize embeddings: {e}",
                details=str(e),
                user_message="Failed to initialize AI embeddings. Please check your API key configuration."
            )
    
    @log_processing_step("Embedding creation")
    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            NumPy array of embeddings
        """
        try:
            if not self.embeddings:
                raise RAGError("Embeddings not initialized")
            
            texts = [doc.page_content for doc in documents]
            embeddings_list = self.embeddings.embed_documents(texts)
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            
            logger.info(f"Created embeddings for {len(documents)} documents: shape {embeddings_array.shape}")
            return embeddings_array
            
        except Exception as e:
            if isinstance(e, RAGError):
                raise
            raise RAGError(
                f"Embedding creation failed: {e}",
                details=str(e),
                user_message="Failed to create document embeddings. Please check your internet connection and API key."
            )
    
    @validate_api_operation
    def embed_query(self, query: str) -> np.ndarray:
        """
        Create embedding for a single query.
        
        Args:
            query: Query string to embed
            
        Returns:
            NumPy array of query embedding
        """
        try:
            if not self.embeddings:
                raise RAGError("Embeddings not initialized")
            
            embedding = self.embeddings.embed_query(query)
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            raise RAGError(
                f"Query embedding failed: {e}",
                details=str(e),
                user_message="Failed to process query for search. Please try again."
            )


class VectorStore:
    """Manages FAISS vector store for document retrieval."""
    
    def __init__(self):
        """Initialize vector store."""
        self.index = None
        self.documents = []
        self.dimension = None
    
    @log_processing_step("Vector store creation")
    def create_index(self, embeddings: np.ndarray, documents: List[Document]):
        """
        Create FAISS index from embeddings and documents.
        
        Args:
            embeddings: NumPy array of embeddings
            documents: List of Document objects
        """
        try:
            if embeddings.shape[0] != len(documents):
                raise RAGError("Mismatch between embeddings and documents count")
            
            self.dimension = embeddings.shape[1]
            self.documents = documents
            
            # Create FAISS index (using L2 distance)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
            
            logger.info(f"Created FAISS index with {len(documents)} documents, dimension {self.dimension}")
            
        except Exception as e:
            if isinstance(e, RAGError):
                raise
            raise RAGError(
                f"Vector store creation failed: {e}",
                details=str(e),
                user_message="Failed to create document search index. Please try again."
            )
    
    @error_handler("Vector search", reraise=True)
    def search(self, query_embedding: np.ndarray, k: int = None) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of tuples (Document, similarity_score)
        """
        if not self.index:
            raise RAGError("Vector store not initialized")
        
        k = k or config.retrieval_k
        k = min(k, len(self.documents))  # Don't search for more than available
        
        # Reshape query for FAISS
        query_vector = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Return documents with similarity scores
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                # Convert L2 distance to similarity score (higher is better)
                similarity = 1 / (1 + distance)
                results.append((self.documents[idx], similarity))
        
        logger.debug(f"Vector search returned {len(results)} results")
        return results
    
    def save_index(self, filepath: str):
        """Save FAISS index and documents to file."""
        try:
            index_path = f"{filepath}.faiss"
            docs_path = f"{filepath}.docs"
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save documents
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info(f"Saved vector store to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
    def load_index(self, filepath: str) -> bool:
        """Load FAISS index and documents from file."""
        try:
            index_path = f"{filepath}.faiss"
            docs_path = f"{filepath}.docs"
            
            if not (Path(index_path).exists() and Path(docs_path).exists()):
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load documents
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            self.dimension = self.index.d
            logger.info(f"Loaded vector store from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False


class RAGProcessor:
    """Main RAG processor that combines all components."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize RAG processor.
        
        Args:
            api_key: Google API key for embeddings
        """
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker()
        self.embedding_manager = EmbeddingManager(api_key)
        self.vector_stores = {}  # Store multiple vector stores by name
    
    @log_processing_step("Complete RAG processing")
    def process_pdf(self, pdf_path: str, source_type: str) -> Dict[str, Any]:
        """
        Process a PDF file through the complete RAG pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            source_type: Type of document (resume, job_description, etc.)
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Extract text
            text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            # Analyze content
            analysis = self.pdf_processor.analyze_pdf_content(text)
            
            # Chunk text
            documents = self.text_chunker.chunk_text(text, source_type)
            
            # Create embeddings
            embeddings = self.embedding_manager.create_embeddings(documents)
            
            # Create vector store
            vector_store = VectorStore()
            vector_store.create_index(embeddings, documents)
            
            # Store vector store
            self.vector_stores[source_type] = vector_store
            
            result = {
                'text': text,
                'analysis': analysis,
                'documents': documents,
                'vector_store': vector_store,
                'embedding_count': len(embeddings)
            }
            
            logger.info(f"Successfully processed {source_type} PDF: {len(documents)} chunks, {len(embeddings)} embeddings")
            return result
            
        except Exception as e:
            if isinstance(e, RAGError):
                raise
            raise RAGError(
                f"RAG processing failed for {source_type}: {e}",
                details=str(e),
                user_message=f"Failed to process {source_type} document. Please check the file and try again."
            )
    
    def get_relevant_context(
        self, 
        query: str, 
        source_types: List[str] = None, 
        k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query from processed documents.
        
        Args:
            query: Query string
            source_types: List of source types to search (if None, search all)
            k: Number of results per source type
            
        Returns:
            List of relevant context dictionaries
        """
        try:
            if not self.vector_stores:
                return []
            
            # Default to all available source types
            if source_types is None:
                source_types = list(self.vector_stores.keys())
            
            # Create query embedding
            query_embedding = self.embedding_manager.embed_query(query)
            
            # Search each vector store
            all_results = []
            for source_type in source_types:
                if source_type in self.vector_stores:
                    vector_store = self.vector_stores[source_type]
                    results = vector_store.search(query_embedding, k)
                    
                    for doc, score in results:
                        all_results.append({
                            'content': doc.page_content,
                            'source_type': source_type,
                            'similarity_score': score,
                            'metadata': doc.metadata
                        })
            
            # Sort by similarity score (descending)
            all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.debug(f"Retrieved {len(all_results)} relevant contexts for query")
            return all_results
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of processed documents."""
        stats = {
            'total_vector_stores': len(self.vector_stores),
            'source_types': list(self.vector_stores.keys()),
            'total_documents': 0,
            'by_source_type': {}
        }
        
        for source_type, vector_store in self.vector_stores.items():
            doc_count = len(vector_store.documents)
            stats['total_documents'] += doc_count
            stats['by_source_type'][source_type] = {
                'document_count': doc_count,
                'vector_dimension': vector_store.dimension
            }
        
        return stats 