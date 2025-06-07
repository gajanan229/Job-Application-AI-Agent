"""
RAG utilities for the Application Factory.

Provides text chunking, embedding generation, vector store creation and management,
and retrieval functions for the Application Factory's RAG-enhanced workflow.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from config.api_keys import get_gemini_api_key

logger = logging.getLogger(__name__)


class RAGManager:
    """
    Manages RAG operations including chunking, embedding, vector store creation,
    and retrieval for the Application Factory.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the RAG manager.
        
        Args:
            api_key (Optional[str]): Google Gemini API key. If None, will try to get from config.
        """
        self.api_key = api_key or get_gemini_api_key()
        if not self.api_key:
            raise ValueError("No API key provided. Please configure Gemini API key.")
        
        # Initialize components
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Cache for loaded vector stores
        self._vector_store_cache: Dict[str, FAISS] = {}
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Chunk text into smaller pieces suitable for embedding.
        
        Args:
            text (str): The text to chunk
            metadata (Optional[Dict[str, Any]]): Metadata to attach to each chunk
            
        Returns:
            List[Document]: List of Document objects with chunked text
        """
        try:
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Create Document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                }
                
                # Add custom metadata if provided
                if metadata:
                    doc_metadata.update(metadata)
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            logger.info(f"Successfully chunked text into {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for better chunking and embedding.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove special characters that might interfere with processing
        # (Keep this minimal to preserve formatting)
        text = text.replace('\r', '')
        
        return text.strip()
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create a FAISS vector store from documents.
        
        Args:
            documents (List[Document]): Documents to embed and store
            
        Returns:
            FAISS: The created vector store
        """
        try:
            if not documents:
                raise ValueError("No documents provided for vector store creation")
            
            logger.info(f"Creating vector store from {len(documents)} documents")
            
            # Create vector store
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            logger.info("Vector store created successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def save_vector_store(self, vector_store: FAISS, save_path) -> bool:
        """
        Save a vector store to disk.
        
        Args:
            vector_store (FAISS): The vector store to save
            save_path: Directory path to save the vector store (str or Path)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert to Path object if it's a string
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save the FAISS index
            vector_store.save_local(str(save_path))
            
            # Save additional metadata
            from datetime import datetime
            metadata = {
                "creation_timestamp": datetime.now().isoformat(),
                "num_documents": vector_store.index.ntotal,
                "embedding_model": "models/embedding-001",
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
            
            metadata_path = save_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Vector store saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            return False
    
    def load_vector_store(self, load_path, allow_dangerous_deserialization: bool = True) -> Optional[FAISS]:
        """
        Load a vector store from disk.
        
        Args:
            load_path: Directory path containing the vector store (str or Path)
            allow_dangerous_deserialization (bool): Whether to allow loading pickled objects
            
        Returns:
            Optional[FAISS]: The loaded vector store, or None if loading failed
        """
        try:
            # Convert to Path object if it's a string
            if isinstance(load_path, str):
                load_path = Path(load_path)
            
            if not load_path.exists():
                logger.warning(f"Vector store path does not exist: {load_path}")
                return None
            
            # Check cache first
            cache_key = str(load_path)
            if cache_key in self._vector_store_cache:
                logger.debug(f"Loading vector store from cache: {load_path}")
                return self._vector_store_cache[cache_key]
            
            # Load from disk
            vector_store = FAISS.load_local(
                folder_path=str(load_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=allow_dangerous_deserialization
            )
            
            # Cache the loaded vector store
            self._vector_store_cache[cache_key] = vector_store
            
            logger.info(f"Vector store loaded from {load_path}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None
    
    def retrieve_similar_chunks(
        self, 
        vector_store: FAISS, 
        query: str, 
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve similar chunks from the vector store.
        
        Args:
            vector_store (FAISS): The vector store to search
            query (str): The query text
            k (int): Number of results to return
            score_threshold (Optional[float]): Minimum similarity score threshold
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        try:
            # Perform similarity search with scores
            results = vector_store.similarity_search_with_score(query, k=k)
            
            # Filter by score threshold if provided
            if score_threshold is not None:
                results = [(doc, score) for doc, score in results if score >= score_threshold]
            
            logger.debug(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving similar chunks: {e}")
            return []
    
    def get_retrieval_context(
        self, 
        vector_store: FAISS, 
        query: str, 
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[str]:
        """
        Get retrieval context as a list of strings.
        
        Args:
            vector_store (FAISS): The vector store to search
            query (str): The query text
            k (int): Number of results to return
            score_threshold (Optional[float]): Minimum similarity score threshold
            
        Returns:
            List[str]: List of relevant text chunks
        """
        try:
            results = self.retrieve_similar_chunks(vector_store, query, k, score_threshold)
            return [doc.page_content for doc, _ in results]
            
        except Exception as e:
            logger.error(f"Error getting retrieval context: {e}")
            return []
    
    def create_section_specific_query(self, job_description: str, section: str) -> str:
        """
        Create a section-specific query for retrieving relevant resume content.
        
        Args:
            job_description (str): The job description
            section (str): The resume section (summary, skills, experience, projects, education)
            
        Returns:
            str: A tailored query for the specific section
        """
        section_queries = {
            "summary": f"Professional summary, career objective, key qualifications, core competencies relevant to: {job_description[:500]}",
            
            "skills": f"Technical skills, programming languages, tools, technologies, software mentioned in: {job_description[:500]}",
            
            "experience": f"Work experience, job responsibilities, achievements, accomplishments related to: {job_description[:500]}",
            
            "projects": f"Projects, software development, technical implementations, solutions relevant to: {job_description[:500]}",
            
            "education": f"Education, degrees, coursework, certifications, academic background relevant to: {job_description[:300]}"
        }
        
        return section_queries.get(section, job_description[:500])
    
    def validate_vector_store(self, vector_store_path) -> bool:
        """
        Validate that a vector store exists and is properly formatted.
        
        Args:
            vector_store_path: Path to the vector store directory (str or Path)
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Convert to Path object if it's a string
            if isinstance(vector_store_path, str):
                vector_store_path = Path(vector_store_path)
            
            if not vector_store_path.exists():
                return False
            
            # Check for required FAISS files
            required_files = ["index.faiss", "index.pkl"]
            for file_name in required_files:
                if not (vector_store_path / file_name).exists():
                    logger.warning(f"Missing required file: {file_name}")
                    return False
            
            # Try to load metadata
            metadata_path = vector_store_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.debug(f"Vector store metadata: {metadata}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating vector store: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear the vector store cache."""
        self._vector_store_cache.clear()
        logger.debug("Vector store cache cleared")
    
    def retrieve_for_resume_section(self, section: str, job_description: str, resume_content: str, k: int = 5) -> List[str]:
        """
        Retrieve context for a specific resume section using temporary vector store.
        
        Args:
            section (str): The resume section (skills, experience, etc.)
            job_description (str): The job description
            resume_content (str): The master resume content
            k (int): Number of chunks to retrieve
            
        Returns:
            List[str]: List of relevant text chunks
        """
        try:
            # Create temporary vector store from resume content
            documents = self.chunk_text(resume_content, {"source": "master_resume"})
            vector_store = self.create_vector_store(documents)
            
            # Create section-specific query
            query = self.create_section_specific_query(job_description, section)
            
            # Retrieve context
            return self.get_retrieval_context(vector_store, query, k)
            
        except Exception as e:
            logger.error(f"Error in retrieve_for_resume_section: {e}")
            return []


# Utility functions for easier access
def chunk_master_resume(resume_content: str, api_key: Optional[str] = None) -> List[Document]:
    """
    Convenience function to chunk a master resume.
    
    Args:
        resume_content (str): The master resume content
        api_key (Optional[str]): API key for embeddings
        
    Returns:
        List[Document]: Chunked resume documents
    """
    rag_manager = RAGManager(api_key)
    metadata = {"source": "master_resume", "document_type": "resume"}
    return rag_manager.chunk_text(resume_content, metadata)


def create_resume_vector_store(resume_content: str, save_path, api_key: Optional[str] = None) -> bool:
    """
    Convenience function to create and save a resume vector store.
    
    Args:
        resume_content (str): The master resume content
        save_path: Path to save the vector store (str or Path)
        api_key (Optional[str]): API key for embeddings
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        rag_manager = RAGManager(api_key)
        
        # Chunk the resume
        documents = chunk_master_resume(resume_content, api_key)
        
        # Create vector store
        vector_store = rag_manager.create_vector_store(documents)
        
        # Save vector store
        return rag_manager.save_vector_store(vector_store, save_path)
        
    except Exception as e:
        logger.error(f"Error creating resume vector store: {e}")
        return False


def retrieve_for_resume_section(
    vector_store_path, 
    job_description: str, 
    section: str,
    k: int = 5,
    api_key: Optional[str] = None
) -> List[str]:
    """
    Convenience function to retrieve context for a specific resume section.
    
    Args:
        vector_store_path: Path to the vector store (str or Path)
        job_description (str): The job description
        section (str): The resume section
        k (int): Number of chunks to retrieve
        api_key (Optional[str]): API key for embeddings
        
    Returns:
        List[str]: List of relevant text chunks
    """
    try:
        rag_manager = RAGManager(api_key)
        
        # Load vector store
        vector_store = rag_manager.load_vector_store(vector_store_path)
        if not vector_store:
            return []
        
        # Create section-specific query
        query = rag_manager.create_section_specific_query(job_description, section)
        
        # Retrieve context
        return rag_manager.get_retrieval_context(vector_store, query, k)
        
    except Exception as e:
        logger.error(f"Error retrieving context for section {section}: {e}")
        return [] 