"""
Test script for Phase 2: RAG Infrastructure & Vector Store Implementation

This script tests the RAG utilities, vector store creation, and retrieval
functionality using the existing Base_Resume.txt file.
"""

import os
import sys
from pathlib import Path
import logging

# Add the current directory to the path so we can import our modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our modules
from rag_utils import RAGManager, create_resume_vector_store, retrieve_for_resume_section
from config.paths import PathManager
from config.api_keys import get_gemini_api_key
from state_rag import (
    create_initial_state, 
    update_vector_store_path,
    update_retrieved_contexts,
    get_rag_status,
    validate_vector_store_in_state
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_api_key():
    """Test that we can get the API key."""
    try:
        api_key = get_gemini_api_key()
        if api_key:
            logger.info("✅ API key successfully retrieved")
            return True
        else:
            logger.error("❌ No API key found")
            return False
    except Exception as e:
        logger.error(f"❌ Error getting API key: {e}")
        return False


def test_path_manager():
    """Test the PathManager with vector store functionality."""
    try:
        # Initialize PathManager
        path_manager = PathManager("test_output")
        
        # Test vector store path creation
        vs_path = path_manager.get_master_resume_vector_store_path()
        logger.info(f"✅ Vector store path: {vs_path}")
        
        # Test custom vector store creation
        custom_vs_path = path_manager.create_vector_store_folder("test_resume_vs")
        logger.info(f"✅ Custom vector store created: {custom_vs_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PathManager test failed: {e}")
        return False


def test_master_resume_loading():
    """Test loading the Base_Resume.txt file."""
    try:
        base_resume_path = Path("Base_Resume.txt")
        
        if not base_resume_path.exists():
            logger.error("❌ Base_Resume.txt not found")
            return False, ""
        
        with open(base_resume_path, 'r', encoding='utf-8') as f:
            resume_content = f.read()
        
        logger.info(f"✅ Master resume loaded: {len(resume_content)} characters")
        return True, resume_content
        
    except Exception as e:
        logger.error(f"❌ Error loading master resume: {e}")
        return False, ""


def test_rag_manager_initialization():
    """Test RAGManager initialization."""
    try:
        api_key = get_gemini_api_key()
        if not api_key:
            logger.error("❌ Cannot test RAGManager without API key")
            return False, None
        
        rag_manager = RAGManager(api_key)
        logger.info("✅ RAGManager initialized successfully")
        return True, rag_manager
        
    except Exception as e:
        logger.error(f"❌ RAGManager initialization failed: {e}")
        return False, None


def test_text_chunking(rag_manager, resume_content):
    """Test text chunking functionality."""
    try:
        documents = rag_manager.chunk_text(
            resume_content, 
            metadata={"source": "test_master_resume", "document_type": "resume"}
        )
        
        logger.info(f"✅ Text chunked into {len(documents)} documents")
        
        # Show a sample chunk
        if documents:
            sample_chunk = documents[0]
            logger.info(f"Sample chunk: {sample_chunk.page_content[:100]}...")
            logger.info(f"Sample metadata: {sample_chunk.metadata}")
        
        return True, documents
        
    except Exception as e:
        logger.error(f"❌ Text chunking failed: {e}")
        return False, None


def test_vector_store_creation(rag_manager, documents):
    """Test vector store creation and saving."""
    try:
        # Create vector store
        vector_store = rag_manager.create_vector_store(documents)
        logger.info("✅ Vector store created successfully")
        
        # Test saving
        path_manager = PathManager("test_output")
        save_path = path_manager.get_master_resume_vector_store_path()
        
        success = rag_manager.save_vector_store(vector_store, save_path)
        if success:
            logger.info(f"✅ Vector store saved to: {save_path}")
        else:
            logger.error("❌ Vector store saving failed")
            return False, None, None
        
        return True, vector_store, save_path
        
    except Exception as e:
        logger.error(f"❌ Vector store creation failed: {e}")
        return False, None, None


def test_vector_store_loading(rag_manager, save_path):
    """Test vector store loading."""
    try:
        loaded_vector_store = rag_manager.load_vector_store(save_path)
        
        if loaded_vector_store:
            logger.info("✅ Vector store loaded successfully")
            return True, loaded_vector_store
        else:
            logger.error("❌ Vector store loading failed")
            return False, None
            
    except Exception as e:
        logger.error(f"❌ Vector store loading failed: {e}")
        return False, None


def test_retrieval_functionality(rag_manager, vector_store):
    """Test retrieval functionality with sample queries."""
    try:
        # Test queries for different resume sections
        test_queries = {
            "skills": "Python programming, machine learning, web development skills",
            "experience": "work experience, research assistant, software development",
            "projects": "software projects, applications, technical implementations",
            "education": "computer science education, university, coursework"
        }
        
        results = {}
        
        for section, query in test_queries.items():
            retrieved_chunks = rag_manager.get_retrieval_context(
                vector_store, query, k=3
            )
            results[section] = retrieved_chunks
            logger.info(f"✅ Retrieved {len(retrieved_chunks)} chunks for '{section}' query")
            
            # Show sample result
            if retrieved_chunks:
                logger.info(f"Sample result for {section}: {retrieved_chunks[0][:100]}...")
        
        return True, results
        
    except Exception as e:
        logger.error(f"❌ Retrieval testing failed: {e}")
        return False, None


def test_state_integration():
    """Test integration with state management."""
    try:
        # Create initial state
        state = create_initial_state(
            master_resume_path="Base_Resume.txt",
            job_description_content="Software Engineer position requiring Python and ML skills",
            output_base_path="test_output"
        )
        
        logger.info("✅ Initial state created")
        
        # Test vector store path update
        path_manager = PathManager("test_output")
        vs_path = str(path_manager.get_master_resume_vector_store_path())
        updated_state = update_vector_store_path(state, vs_path)
        
        logger.info("✅ State updated with vector store path")
        
        # Test RAG status
        rag_status = get_rag_status(updated_state)
        logger.info(f"RAG Status: {rag_status}")
        
        # Test vector store validation in state
        # Note: This will fail if vector store doesn't exist yet, which is expected
        is_valid = validate_vector_store_in_state(updated_state)
        logger.info(f"Vector store valid in state: {is_valid}")
        
        return True, updated_state
        
    except Exception as e:
        logger.error(f"❌ State integration test failed: {e}")
        return False, None


def test_convenience_functions():
    """Test the convenience functions."""
    try:
        # Test create_resume_vector_store convenience function
        success, resume_content = test_master_resume_loading()
        if not success:
            return False
        
        path_manager = PathManager("test_output")
        save_path = path_manager.create_vector_store_folder("convenience_test")
        
        success = create_resume_vector_store(resume_content, save_path)
        if success:
            logger.info("✅ Convenience function create_resume_vector_store works")
        else:
            logger.error("❌ Convenience function failed")
            return False
        
        # Test retrieve_for_resume_section convenience function
        test_job_description = "Software Engineer position requiring Python, machine learning, and web development skills"
        
        retrieved_context = retrieve_for_resume_section(
            save_path, 
            test_job_description, 
            "skills",
            k=3
        )
        
        if retrieved_context:
            logger.info(f"✅ Convenience function retrieve_for_resume_section works: {len(retrieved_context)} chunks")
            logger.info(f"Sample retrieved context: {retrieved_context[0][:100]}...")
        else:
            logger.error("❌ Retrieval convenience function failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Convenience functions test failed: {e}")
        return False


def main():
    """Run all Phase 2 tests."""
    logger.info("🚀 Starting Phase 2: RAG Infrastructure Tests")
    logger.info("=" * 50)
    
    # Test results tracking
    test_results = {}
    
    # Test 1: API Key
    logger.info("\n1. Testing API Key Configuration...")
    test_results["api_key"] = test_api_key()
    
    # Test 2: PathManager
    logger.info("\n2. Testing PathManager...")
    test_results["path_manager"] = test_path_manager()
    
    # Test 3: Master Resume Loading
    logger.info("\n3. Testing Master Resume Loading...")
    success, resume_content = test_master_resume_loading()
    test_results["resume_loading"] = success
    
    if not success:
        logger.error("Cannot continue without master resume. Stopping tests.")
        return
    
    # Test 4: RAGManager Initialization
    logger.info("\n4. Testing RAGManager Initialization...")
    success, rag_manager = test_rag_manager_initialization()
    test_results["rag_manager_init"] = success
    
    if not success:
        logger.error("Cannot continue without RAGManager. Stopping tests.")
        return
    
    # Test 5: Text Chunking
    logger.info("\n5. Testing Text Chunking...")
    success, documents = test_text_chunking(rag_manager, resume_content)
    test_results["text_chunking"] = success
    
    if not success:
        logger.error("Cannot continue without text chunking. Stopping tests.")
        return
    
    # Test 6: Vector Store Creation
    logger.info("\n6. Testing Vector Store Creation...")
    success, vector_store, save_path = test_vector_store_creation(rag_manager, documents)
    test_results["vector_store_creation"] = success
    
    if not success:
        logger.error("Cannot continue without vector store. Stopping tests.")
        return
    
    # Test 7: Vector Store Loading
    logger.info("\n7. Testing Vector Store Loading...")
    success, loaded_vector_store = test_vector_store_loading(rag_manager, save_path)
    test_results["vector_store_loading"] = success
    
    # Test 8: Retrieval Functionality
    logger.info("\n8. Testing Retrieval Functionality...")
    success, retrieval_results = test_retrieval_functionality(rag_manager, loaded_vector_store or vector_store)
    test_results["retrieval"] = success
    
    # Test 9: State Integration
    logger.info("\n9. Testing State Integration...")
    success, state = test_state_integration()
    test_results["state_integration"] = success
    
    # Test 10: Convenience Functions
    logger.info("\n10. Testing Convenience Functions...")
    test_results["convenience_functions"] = test_convenience_functions()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 PHASE 2 TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Phase 2 tests passed! RAG infrastructure is ready.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Review and fix issues before proceeding.")


if __name__ == "__main__":
    main() 