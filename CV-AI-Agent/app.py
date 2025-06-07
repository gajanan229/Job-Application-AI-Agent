"""
Main Streamlit application for the Application Factory.
"""

import streamlit as st
from pathlib import Path

# Import our foundation modules
from config.settings import config
from config.logging_config import setup_streamlit_logging, get_logger
from utils.session_utils import SessionManager
from utils.error_handlers import show_error_messages, show_success_messages, streamlit_error_handler
from utils.file_utils import FileManager

# Import Phase 2 RAG modules
from core.rag_processor import RAGProcessor

# Import Phase 3 LLM modules
from core.llm_service import ContentGenerator, ContentType

# Import Phase 5 LangGraph Workflow Engine
from core.graph import ApplicationFactoryWorkflow, WorkflowStatus

# Import content validation utilities
from utils.content_validators import get_content_metrics

# Initialize logging
logger = setup_streamlit_logging()

def main():
    """Main application function."""
    
    # Page configuration
    st.set_page_config(
        page_title="Application Factory",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    SessionManager.initialize_session()
    
    # Sidebar for API key configuration
    setup_sidebar()
    
    # Header
    st.title("🏭 Application Factory")
    st.markdown("*Your Personal Document Crafting Studio*")
    
    # Show any messages
    show_error_messages()
    show_success_messages()
    
    # Determine which page to show based on current stage
    current_stage = SessionManager.get_current_stage()
    
    if current_stage == "setup":
        display_setup_page()
    elif current_stage == "workflow_processing":
        display_workflow_processing_page()
    elif current_stage == "workflow_complete":
        display_workflow_results_page()
    elif current_stage == "rag_processing":
        display_rag_processing_page()
    elif current_stage == "rag_complete":
        display_rag_results_page()
    elif current_stage == "content_generation":
        display_content_generation_page()
    elif current_stage == "generation_processing":
        display_generation_processing()
    elif current_stage == "generation_complete":
        display_generation_results_page()
    else:
        display_setup_page()


def setup_sidebar():
    """Setup sidebar with API key configuration and navigation."""
    st.sidebar.title("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "Google API Key",
        type="password",
        help="Enter your Google API key for Gemini embeddings"
    )
    
    if api_key:
        # Update session state with API key
        SessionManager.set_app_state_value("api_key", api_key)
        st.sidebar.success("✅ API Key configured")
    else:
        st.sidebar.warning("⚠️ API Key required for document processing")
    
    # Session controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Controls")
    
    if st.sidebar.button("🔄 Reset Session"):
        SessionManager.clear_session()
        st.rerun()
    
    if st.sidebar.button("🧹 Cleanup Temp Files"):
        cleaned = FileManager.cleanup_temp_files(0)
        SessionManager.add_success_message(f"Cleaned {cleaned} temporary files")
        st.rerun()
    
    # Debug information
    if config.debug:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🐛 Debug Info")
        session_info = SessionManager.get_session_info()
        st.sidebar.json(session_info)


@streamlit_error_handler
def display_setup_page():
    """Display the setup page with separate file upload sections."""
    
    st.header("📤 Document Upload")
    st.markdown("Upload your master resume and job description to get started.")
    
    # Check if API key is configured
    api_key = SessionManager.get_app_state_value("api_key")
    if not api_key:
        st.warning("⚠️ Please configure your Google API key in the sidebar before uploading documents.")
        return
    
    # Create two columns for file uploads
    col1, col2 = st.columns(2)
    
    # Master Resume Upload
    with col1:
        st.subheader("📄 Master Resume")
        st.markdown("Upload your comprehensive master resume (PDF)")
        
        master_resume_file = st.file_uploader(
            "Choose your master resume PDF",
            type="pdf",
            key="master_resume_upload",
            help="This should be your complete resume with all experiences, projects, and skills"
        )
        
        if master_resume_file is not None:
            # Validate file
            is_valid, error_message = FileManager.validate_pdf(master_resume_file)
            
            if is_valid:
                st.success("✅ Master resume validated successfully!")
                
                # Show file info
                file_info = {
                    "name": master_resume_file.name,
                    "size": f"{master_resume_file.size / 1024:.1f} KB"
                }
                st.info(f"**File:** {file_info['name']} ({file_info['size']})")
                
                # Save and process button
                if st.button("💾 Save Master Resume", key="save_master_resume"):
                    try:
                        # Save file
                        file_path = FileManager.save_uploaded_file(master_resume_file, "master_resume")
                        SessionManager.add_temp_file(file_path)
                        SessionManager.set_app_state_value("master_resume_pdf_path", file_path)
                        SessionManager.add_success_message("Master resume saved successfully!")
                        st.rerun()
                    except Exception as e:
                        SessionManager.add_error_message(f"Failed to save master resume: {str(e)}")
                        st.rerun()
            else:
                st.error(f"❌ Validation failed: {error_message}")
    
    # Job Description Upload
    with col2:
        st.subheader("💼 Job Description")
        st.markdown("Upload the job description you're targeting (PDF)")
        
        job_desc_file = st.file_uploader(
            "Choose job description PDF",
            type="pdf",
            key="job_desc_upload",
            help="Upload the job posting or description for the position you're applying to"
        )
        
        if job_desc_file is not None:
            # Validate file
            is_valid, error_message = FileManager.validate_pdf(job_desc_file)
            
            if is_valid:
                st.success("✅ Job description validated successfully!")
                
                # Show file info
                file_info = {
                    "name": job_desc_file.name,
                    "size": f"{job_desc_file.size / 1024:.1f} KB"
                }
                st.info(f"**File:** {file_info['name']} ({file_info['size']})")
                
                # Save and process button
                if st.button("💾 Save Job Description", key="save_job_desc"):
                    try:
                        # Save file
                        file_path = FileManager.save_uploaded_file(job_desc_file, "job_description")
                        SessionManager.add_temp_file(file_path)
                        SessionManager.set_app_state_value("job_description_pdf_path", file_path)
                        SessionManager.add_success_message("Job description saved successfully!")
                        st.rerun()
                    except Exception as e:
                        SessionManager.add_error_message(f"Failed to save job description: {str(e)}")
                        st.rerun()
            else:
                st.error(f"❌ Validation failed: {error_message}")
    
    # Check if both files are uploaded
    master_resume_path = SessionManager.get_app_state_value("master_resume_pdf_path")
    job_desc_path = SessionManager.get_app_state_value("job_description_pdf_path")
    
    st.markdown("---")
    
    # Show upload status
    st.subheader("📋 Upload Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        if master_resume_path:
            st.success("✅ Master Resume")
            st.caption(f"📁 {Path(master_resume_path).name}")
        else:
            st.error("❌ Master Resume")
            st.caption("Not uploaded")
    
    with status_col2:
        if job_desc_path:
            st.success("✅ Job Description")
            st.caption(f"📁 {Path(job_desc_path).name}")
        else:
            st.error("❌ Job Description")
            st.caption("Not uploaded")
    
    with status_col3:
        if master_resume_path and job_desc_path:
            st.success("✅ Ready to Process")
            st.caption("Both documents uploaded")
        else:
            st.warning("⏳ Waiting for uploads")
            st.caption("Upload both documents")
    
    # Process documents button
    if master_resume_path and job_desc_path:
        st.markdown("---")
        
        # Option to use workflow engine or manual processing
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 Start AI Workflow", type="primary", use_container_width=True):
                SessionManager.set_current_stage("workflow_processing")
                SessionManager.set_processing_status("Initializing AI workflow engine...")
                st.rerun()
        
        with col2:
            if st.button("⚙️ Manual Processing", use_container_width=True):
                SessionManager.set_current_stage("rag_processing")
                SessionManager.set_processing_status("Initializing manual processing...")
                st.rerun()
        
        st.info("💡 **AI Workflow**: Automated end-to-end processing using LangGraph  \n**Manual Processing**: Step-by-step processing with user control")


@streamlit_error_handler
def display_rag_processing_page():
    """Display the RAG processing page with progress indicators."""
    
    st.header("🔄 Processing Documents")
    st.markdown("Analyzing your documents using AI-powered RAG pipeline...")
    
    # Get file paths
    master_resume_path = SessionManager.get_app_state_value("master_resume_pdf_path")
    job_desc_path = SessionManager.get_app_state_value("job_description_pdf_path")
    api_key = SessionManager.get_app_state_value("api_key")
    
    if not all([master_resume_path, job_desc_path, api_key]):
        st.error("Missing required files or API key. Returning to setup...")
        SessionManager.set_current_stage("setup")
        st.rerun()
        return
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize RAG processor
        status_text.text("🔧 Initializing RAG processor...")
        progress_bar.progress(10)
        
        rag_processor = RAGProcessor(api_key=api_key)
        SessionManager.set_app_state_value("rag_processor", rag_processor)
        
        # Process master resume
        status_text.text("📄 Processing master resume...")
        progress_bar.progress(30)
        
        master_result = rag_processor.process_pdf(master_resume_path, "master_resume")
        SessionManager.set_app_state_value("master_resume_result", master_result)
        
        # Process job description
        status_text.text("💼 Processing job description...")
        progress_bar.progress(70)
        
        job_desc_result = rag_processor.process_pdf(job_desc_path, "job_description")
        SessionManager.set_app_state_value("job_description_result", job_desc_result)
        
        # Complete processing
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        # Update session state
        SessionManager.set_current_stage("rag_complete")
        SessionManager.set_processing_status(None)
        SessionManager.add_success_message("Document processing completed successfully!")
        
        # Auto-advance after a brief pause
        st.balloons()
        st.success("🎉 Processing completed! Advancing to results...")
        
        # Small delay before advancing
        import time
        time.sleep(2)
        st.rerun()
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ Processing failed")
        SessionManager.add_error_message(f"RAG processing failed: {str(e)}")
        SessionManager.set_current_stage("setup")
        st.rerun()


@streamlit_error_handler
def display_rag_results_page():
    """Display the RAG processing results and analysis."""
    
    st.header("📊 Document Analysis Results")
    st.markdown("Your documents have been processed and are ready for AI-powered content generation.")
    
    # Get results from session state
    master_result = SessionManager.get_app_state_value("master_resume_result")
    job_desc_result = SessionManager.get_app_state_value("job_description_result")
    rag_processor = SessionManager.get_app_state_value("rag_processor")
    
    if not all([master_result, job_desc_result, rag_processor]):
        st.error("Missing processing results. Returning to setup...")
        SessionManager.set_current_stage("setup")
        st.rerun()
        return
    
    # Summary statistics
    st.subheader("📈 Processing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Resume Chunks",
            len(master_result['documents']),
            help="Number of text chunks created from your resume"
        )
    
    with col2:
        st.metric(
            "Job Desc Chunks",
            len(job_desc_result['documents']),
            help="Number of text chunks created from job description"
        )
    
    with col3:
        st.metric(
            "Total Embeddings",
            master_result['embedding_count'] + job_desc_result['embedding_count'],
            help="Total AI embeddings created for semantic search"
        )
    
    with col4:
        vector_stats = rag_processor.get_summary_statistics()
        st.metric(
            "Vector Stores",
            vector_stats['total_vector_stores'],
            help="Number of searchable document databases created"
        )
    
    # Document analysis
    st.subheader("📋 Document Analysis")
    
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        st.markdown("**📄 Master Resume Analysis**")
        resume_analysis = master_result['analysis']
        
        st.write(f"**Words:** {resume_analysis['total_words']:,}")
        st.write(f"**Characters:** {resume_analysis['total_characters']:,}")
        st.write(f"**Estimated Pages:** {resume_analysis['estimated_pages']}")
        st.write(f"**Has Email:** {'✅' if resume_analysis['has_email'] else '❌'}")
        st.write(f"**Has Phone:** {'✅' if resume_analysis['has_phone'] else '❌'}")
        st.write(f"**Has Bullets:** {'✅' if resume_analysis['has_bullets'] else '❌'}")
    
    with analysis_col2:
        st.markdown("**💼 Job Description Analysis**")
        job_analysis = job_desc_result['analysis']
        
        st.write(f"**Words:** {job_analysis['total_words']:,}")
        st.write(f"**Characters:** {job_analysis['total_characters']:,}")
        st.write(f"**Estimated Pages:** {job_analysis['estimated_pages']}")
        st.write(f"**Has Email:** {'✅' if job_analysis['has_email'] else '❌'}")
        st.write(f"**Has Phone:** {'✅' if job_analysis['has_phone'] else '❌'}")
        st.write(f"**Has Bullets:** {'✅' if job_analysis['has_bullets'] else '❌'}")
    
    # Test retrieval functionality
    st.subheader("🔍 Test Document Retrieval")
    st.markdown("Test the RAG system by searching for relevant content:")
    
    test_query = st.text_input(
        "Enter a search query:",
        placeholder="e.g., Python programming, project management, technical skills",
        help="Search for relevant content across both documents"
    )
    
    if test_query:
        if st.button("🔍 Search Documents"):
            try:
                # Get relevant context
                results = rag_processor.get_relevant_context(
                    query=test_query,
                    k=3  # Get top 3 results per document type
                )
                
                if results:
                    st.markdown("**🎯 Search Results:**")
                    
                    for i, result in enumerate(results[:6]):  # Show top 6 results
                        with st.expander(
                            f"Result {i+1}: {result['source_type']} (Score: {result['similarity_score']:.3f})"
                        ):
                            st.write(result['content'])
                            st.caption(f"Source: {result['source_type']} | Chunk: {result['metadata']['chunk_index']}")
                else:
                    st.info("No relevant results found for your query.")
                    
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
    
    # Navigation buttons
    st.markdown("---")
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        if st.button("🔄 Process New Documents"):
            SessionManager.set_current_stage("setup")
            st.rerun()
    
    with nav_col2:
        if st.button("🚀 Generate Content"):
            SessionManager.set_current_stage("content_generation")
            st.rerun()
    
    with nav_col3:
        if st.button("📊 View Analytics"):
            st.info("Analytics dashboard coming soon!")
    
    # Advanced debugging
    if config.debug:
        st.markdown("---")
        st.subheader("🐛 Advanced Debug Information")
        
        with st.expander("Vector Store Statistics"):
            st.json(rag_processor.get_summary_statistics())
        
        with st.expander("Master Resume Sample Chunks"):
            if master_result['documents']:
                for i, doc in enumerate(master_result['documents'][:3]):
                    st.write(f"**Chunk {i}:**")
                    st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
        
        with st.expander("Job Description Sample Chunks"):
            if job_desc_result['documents']:
                for i, doc in enumerate(job_desc_result['documents'][:3]):
                    st.write(f"**Chunk {i}:**")
                    st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)


@streamlit_error_handler
def display_content_generation_page():
    """Display the content generation page with options and controls."""
    
    st.header("🚀 AI Content Generation")
    st.markdown("Generate tailored resumes and cover letters using advanced AI and RAG technology.")
    
    # Check prerequisites
    api_key = SessionManager.get_app_state_value("api_key")
    rag_processor = SessionManager.get_app_state_value("rag_processor")
    master_result = SessionManager.get_app_state_value("master_resume_result")
    job_desc_result = SessionManager.get_app_state_value("job_description_result")
    
    if not all([api_key, rag_processor, master_result, job_desc_result]):
        st.error("Missing required data. Returning to setup...")
        SessionManager.set_current_stage("setup")
        st.rerun()
        return
    
    # Content generation options
    st.subheader("📋 Generation Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📝 Content Types**")
        generate_resume = st.checkbox("Resume", value=True, help="Generate tailored resume")
        generate_cover_letter = st.checkbox("Cover Letter", value=True, help="Generate cover letter")
        generate_analysis = st.checkbox("Document Analysis", value=False, help="Generate document alignment analysis")
    
    with col2:
        st.markdown("**⚙️ Preferences**")
        tone = st.selectbox(
            "Tone Style",
            ["Professional", "Conversational", "Dynamic", "Conservative"],
            help="Choose the overall tone for generated content"
        )
        
        focus_area = st.selectbox(
            "Focus Area",
            ["Technical Skills", "Leadership", "Project Management", "Results & Achievements", "Balanced"],
            help="Primary focus area for content optimization"
        )
        
        additional_instructions = st.text_area(
            "Additional Instructions",
            placeholder="Any specific requirements or preferences for the generated content...",
            help="Optional: Provide specific instructions for content generation"
        )
    
    # Get content types to generate
    content_types = []
    if generate_resume:
        content_types.append(ContentType.RESUME)
    if generate_cover_letter:
        content_types.append(ContentType.COVER_LETTER)
    if generate_analysis:
        content_types.append(ContentType.ANALYSIS)
    
    if not content_types:
        st.warning("⚠️ Please select at least one content type to generate.")
        return
    
    # Generation controls
    st.markdown("---")
    st.subheader("🎯 Generate Content")
    
    gen_col1, gen_col2, gen_col3 = st.columns(3)
    
    with gen_col1:
        if st.button("🚀 Start Generation", type="primary", use_container_width=True):
            # Prepare user preferences
            user_preferences = {
                "tone": tone.lower(),
                "focus_area": focus_area.lower().replace(" ", "_"),
                "additional_instructions": additional_instructions
            }
            
            # Store generation parameters in session
            SessionManager.set_app_state_value("generation_content_types", content_types)
            SessionManager.set_app_state_value("generation_preferences", user_preferences)
            
            # Start generation process
            SessionManager.set_current_stage("generation_processing")
            SessionManager.set_processing_status("Initializing content generation...")
            st.rerun()
    
    with gen_col2:
        if st.button("🔙 Back to Results", use_container_width=True):
            SessionManager.set_current_stage("rag_complete")
            st.rerun()
    
    with gen_col3:
        if st.button("🏠 Start Over", use_container_width=True):
            SessionManager.set_current_stage("setup")
            st.rerun()
    
    # Show generation preview
    st.markdown("---")
    st.subheader("📊 Generation Preview")
    
    preview_col1, preview_col2, preview_col3 = st.columns(3)
    
    with preview_col1:
        st.metric("Content Types", len(content_types))
        if content_types:
            for content_type in content_types:
                st.caption(f"• {content_type.value.replace('_', ' ').title()}")
    
    with preview_col2:
        st.metric("RAG Context Available", "Yes" if rag_processor else "No")
        st.caption(f"• Resume chunks: {len(master_result['documents'])}")
        st.caption(f"• Job desc chunks: {len(job_desc_result['documents'])}")
    
    with preview_col3:
        st.metric("Personalization", "High" if additional_instructions else "Standard")
        st.caption(f"• Tone: {tone}")
        st.caption(f"• Focus: {focus_area}")


@streamlit_error_handler
def display_generation_processing():
    """Handle the actual content generation process."""
    
    st.header("⚡ Generating Content...")
    st.markdown("Please wait while AI crafts your personalized documents.")
    
    # Get generation parameters
    api_key = SessionManager.get_app_state_value("api_key")
    rag_processor = SessionManager.get_app_state_value("rag_processor")
    master_result = SessionManager.get_app_state_value("master_resume_result")
    job_desc_result = SessionManager.get_app_state_value("job_description_result")
    content_types = SessionManager.get_app_state_value("generation_content_types")
    user_preferences = SessionManager.get_app_state_value("generation_preferences")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize content generator
        status_text.text("🔧 Initializing AI content generator...")
        progress_bar.progress(10)
        
        content_generator = ContentGenerator(api_key=api_key)
        
        # Extract text content from RAG results
        status_text.text("📄 Preparing document content...")
        progress_bar.progress(20)
        
        master_resume_text = "\n\n".join([doc.page_content for doc in master_result['documents']])
        job_description_text = "\n\n".join([doc.page_content for doc in job_desc_result['documents']])
        
        # Get relevant RAG context
        status_text.text("🔍 Retrieving relevant context...")
        progress_bar.progress(30)
        
        # Use a comprehensive query to get relevant context
        context_query = f"skills experience qualifications requirements {user_preferences.get('focus_area', '').replace('_', ' ')}"
        rag_context = rag_processor.get_relevant_context(query=context_query, k=5)
        
        # Generate content
        status_text.text("🤖 Generating AI content...")
        progress_bar.progress(50)
        
        generation_results = content_generator.batch_generate(
            job_description=job_description_text,
            master_resume_text=master_resume_text,
            rag_context=rag_context,
            content_types=content_types,
            user_preferences=user_preferences
        )
        
        # Store results
        status_text.text("💾 Saving generated content...")
        progress_bar.progress(90)
        
        SessionManager.set_app_state_value("generation_results", generation_results)
        SessionManager.set_app_state_value("generation_context", rag_context)
        
        # Complete
        status_text.text("✅ Content generation complete!")
        progress_bar.progress(100)
        
        # Update session state
        SessionManager.set_current_stage("generation_complete")
        SessionManager.set_processing_status(None)
        SessionManager.add_success_message("Content generation completed successfully!")
        
        # Auto-advance
        st.balloons()
        st.success("🎉 Generation completed! Advancing to results...")
        
        import time
        time.sleep(2)
        st.rerun()
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ Generation failed")
        SessionManager.add_error_message(f"Content generation failed: {str(e)}")
        SessionManager.set_current_stage("content_generation")
        st.rerun()


@streamlit_error_handler
def display_generation_results_page():
    """Display the generated content results."""
    
    st.header("📄 Generated Content")
    st.markdown("Your AI-crafted documents are ready! Review, edit, and download below.")
    
    # Get results from session
    generation_results = SessionManager.get_app_state_value("generation_results")
    generation_context = SessionManager.get_app_state_value("generation_context")
    
    if not generation_results:
        st.error("No generation results found. Returning to content generation...")
        SessionManager.set_current_stage("content_generation")
        st.rerun()
        return
    
    # Summary metrics
    st.subheader("📊 Generation Summary")
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric("Documents Generated", len(generation_results))
    
    with metric_cols[1]:
        total_time = sum(result.generation_time for result in generation_results.values())
        st.metric("Total Generation Time", f"{total_time:.1f}s")
    
    with metric_cols[2]:
        total_tokens = sum(
            result.token_usage.get('total_tokens', 0) 
            for result in generation_results.values() 
            if result.token_usage
        )
        st.metric("Tokens Used", f"{total_tokens:,}" if total_tokens > 0 else "N/A")
    
    with metric_cols[3]:
        context_chunks = len(generation_context) if generation_context else 0
        st.metric("Context Chunks Used", context_chunks)
    
    # Display generated content
    st.markdown("---")
    
    for content_type, result in generation_results.items():
        
        # Content type header
        content_title = content_type.value.replace('_', ' ').title()
        
        if content_type == ContentType.RESUME:
            icon = "📄"
        elif content_type == ContentType.COVER_LETTER:
            icon = "💌"
        elif content_type == ContentType.ANALYSIS:
            icon = "📊"
        else:
            icon = "📝"
        
        st.subheader(f"{icon} {content_title}")
        
        # Content display with editing capability
        edited_content = st.text_area(
            f"Edit {content_title}",
            value=result.content,
            height=400,
            key=f"edit_{content_type.value}",
            help=f"Review and edit your generated {content_title.lower()}"
        )
        
        # Update the result with edited content if changed
        if edited_content != result.content:
            result.content = edited_content
            SessionManager.set_app_state_value("generation_results", generation_results)
        
        # Generation metadata
        with st.expander(f"📈 {content_title} Generation Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Generation Time:** {result.generation_time:.2f} seconds")
                st.write(f"**Model:** {result.metadata.get('model', 'Unknown')}")
                st.write(f"**Temperature:** {result.metadata.get('temperature', 'Unknown')}")
            
            with col2:
                st.write(f"**Content Length:** {len(result.content):,} characters")
                st.write(f"**RAG Chunks Used:** {result.metadata.get('rag_chunks_used', 0)}")
                if result.token_usage:
                    st.write(f"**Tokens Used:** {result.token_usage.get('total_tokens', 0):,}")
        
        # Download button
        if st.button(f"💾 Save {content_title} as Text", key=f"save_{content_type.value}"):
            try:
                filename = f"{content_type.value}_{SessionManager.get_timestamp()}.txt"
                file_path = FileManager.save_text_content(result.content, filename)
                SessionManager.add_success_message(f"{content_title} saved as {filename}")
                st.success(f"✅ {content_title} saved successfully!")
            except Exception as e:
                SessionManager.add_error_message(f"Failed to save {content_title}: {str(e)}")
        
        st.markdown("---")
    
    # Navigation controls
    st.subheader("🚀 Next Steps")
    
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        if st.button("🔄 Generate Again", use_container_width=True):
            SessionManager.set_current_stage("content_generation")
            st.rerun()
    
    with nav_col2:
        if st.button("📊 View RAG Results", use_container_width=True):
            SessionManager.set_current_stage("rag_complete")
            st.rerun()
    
    with nav_col3:
        if st.button("🏠 Start Over", use_container_width=True):
            SessionManager.set_current_stage("setup")
            st.rerun()
    
    with nav_col4:
        if st.button("📋 Export All", use_container_width=True):
            try:
                # Save all generated content
                timestamp = SessionManager.get_timestamp()
                saved_files = []
                
                for content_type, result in generation_results.items():
                    filename = f"{content_type.value}_{timestamp}.txt"
                    file_path = FileManager.save_text_content(result.content, filename)
                    saved_files.append(filename)
                
                SessionManager.add_success_message(f"All content exported! Files: {', '.join(saved_files)}")
                st.success("✅ All content exported successfully!")
                
            except Exception as e:
                SessionManager.add_error_message(f"Export failed: {str(e)}")
    
    # RAG context used
    if generation_context and config.debug:
        st.markdown("---")
        st.subheader("🔍 RAG Context Used")
        
        with st.expander("View Context Chunks"):
            for i, context in enumerate(generation_context[:5]):
                st.write(f"**Chunk {i+1}** (Score: {context.get('similarity_score', 0):.3f})")
                st.write(f"Source: {context.get('source_type', 'Unknown')}")
                st.text(context.get('content', '')[:300] + "..." if len(context.get('content', '')) > 300 else context.get('content', ''))
                st.markdown("---")


@streamlit_error_handler
def display_workflow_processing_page():
    """Display the LangGraph workflow processing page with live progress tracking."""
    
    st.header("🤖 AI Workflow Engine")
    st.markdown("Processing your documents through our intelligent LangGraph workflow...")
    
    # Get file paths and API key
    master_resume_path = SessionManager.get_app_state_value("master_resume_pdf_path")
    job_desc_path = SessionManager.get_app_state_value("job_description_pdf_path")
    api_key = SessionManager.get_app_state_value("api_key")
    
    if not all([master_resume_path, job_desc_path, api_key]):
        st.error("Missing required files or API key. Returning to setup...")
        SessionManager.set_current_stage("setup")
        st.rerun()
        return
    
    # Initialize workflow engine if not already done
    workflow_engine = SessionManager.get_workflow_engine()
    if workflow_engine is None:
        try:
            st.info("🔧 Initializing LangGraph workflow engine...")
            workflow_engine = SessionManager.initialize_workflow_engine()
        except Exception as e:
            st.error(f"Failed to initialize workflow engine: {str(e)}")
            SessionManager.add_error_message(f"Workflow initialization failed: {str(e)}")
            SessionManager.set_current_stage("setup")
            st.rerun()
            return
    
    # Initialize workflow state
    workflow_state = SessionManager.get_workflow_state()
    thread_id = SessionManager.get_workflow_thread_id()
    
    # Create initial state if needed
    if workflow_state is None:
        try:
            # Start workflow execution using the execute_workflow method
            with st.spinner("Starting workflow execution..."):
                final_state = workflow_engine.execute_workflow(
                    api_key=api_key,
                    master_resume_path=master_resume_path,
                    job_description_path=job_desc_path,
                    user_preferences=SessionManager.get_app_state_value("user_preferences", {}),
                    session_id=SessionManager.get_timestamp()
                )
                
                # Store the final state
                SessionManager.set_workflow_state(final_state)
                
            st.success("✅ Workflow completed successfully!")
            
            # Check if workflow completed successfully
            if final_state["workflow_progress"].status == WorkflowStatus.COMPLETED:
                SessionManager.set_current_stage("workflow_complete")
                st.rerun()
            else:
                # Handle workflow failure
                error_msg = final_state.get("errors", ["Unknown error occurred"])[0]
                st.error(f"Workflow failed: {error_msg}")
                SessionManager.add_error_message(f"Workflow failed: {error_msg}")
            
        except Exception as e:
            st.error(f"Failed to start workflow: {str(e)}")
            SessionManager.add_error_message(f"Workflow start failed: {str(e)}")
            SessionManager.set_current_stage("setup")
            st.rerun()
            return
    
    # Display workflow completion status
    workflow_state = SessionManager.get_workflow_state()
    if workflow_state:
        progress = workflow_state["workflow_progress"]
        status = progress.status
        
        st.subheader("📊 Workflow Status")
        
        if status == WorkflowStatus.COMPLETED:
            st.success("🎉 Workflow completed successfully!")
            SessionManager.set_current_stage("workflow_complete")
            st.rerun()
        elif status == WorkflowStatus.FAILED:
            error_msg = progress.error_message or "Unknown error occurred"
            st.error(f"❌ Workflow failed: {error_msg}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Retry Workflow"):
                    SessionManager.set_app_state_value("workflow_state", None)
                    st.rerun()
            
            with col2:
                if st.button("🏠 Back to Setup"):
                    SessionManager.set_current_stage("setup")
                    st.rerun()
        else:
            # Show progress information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                completed_count = len(progress.completed_nodes)
                st.metric("Completed Nodes", f"{completed_count}/{progress.total_nodes}")
            
            with col2:
                st.metric("Current Node", progress.current_node)
            
            with col3:
                st.metric("Status", status.value)
            
            # Progress bar
            progress_bar = st.progress(progress.progress_percentage / 100.0)
            
            # Show logs
            if workflow_state.get("logs"):
                with st.expander("📋 Workflow Logs"):
                    for log in workflow_state["logs"]:
                        st.text(log)
    else:
        st.warning("No workflow state found.")
        if st.button("🏠 Back to Setup"):
            SessionManager.set_current_stage("setup")
            st.rerun()


@streamlit_error_handler 
def display_workflow_results_page():
    """Display the results from the completed LangGraph workflow."""
    
    st.header("🎉 Workflow Complete!")
    st.markdown("Your documents have been processed through our AI workflow engine.")
    
    # Get workflow results
    workflow_state = SessionManager.get_workflow_state()
    
    if not workflow_state or workflow_state["workflow_progress"].status != WorkflowStatus.COMPLETED:
        st.error("No completed workflow results found. Returning to setup...")
        SessionManager.set_current_stage("setup") 
        st.rerun()
        return
    
    # Extract results
    progress = workflow_state["workflow_progress"]
    
    # Workflow summary
    st.subheader("📊 Workflow Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if progress.start_time and progress.end_time:
            total_time = (progress.end_time - progress.start_time).total_seconds()
        else:
            total_time = 0
        st.metric("Total Time", f"{total_time:.1f}s")
    
    with col2:
        documents_processed = 2  # Master resume + job description
        st.metric("Documents Processed", documents_processed)
    
    with col3:
        # Try to get chunking info from RAG results
        master_result = workflow_state.get("master_resume_result", {})
        job_result = workflow_state.get("job_description_result", {})
        total_chunks = 0
        if master_result.get("documents"):
            total_chunks += len(master_result["documents"])
        if job_result.get("documents"):
            total_chunks += len(job_result["documents"])
        st.metric("Text Chunks Created", total_chunks)
    
    with col4:
        embeddings_created = 0
        if master_result.get("embedding_count"):
            embeddings_created += master_result["embedding_count"]
        if job_result.get("embedding_count"):
            embeddings_created += job_result["embedding_count"]
        st.metric("Embeddings Created", embeddings_created)
    
    # Processing stages completed
    st.subheader("✅ Completed Stages")
    
    stages = [
        ("📄", "Document Processing", "PDFs loaded and text extracted"),
        ("🔍", "RAG Pipeline", "Text chunked and embedded for semantic search"),
        ("🤖", "Content Generation", "Resume and cover letter generated using AI"),
        ("📝", "Document Creation", "DOCX files created with professional formatting"),
        ("🎯", "Workflow Finalization", "All outputs prepared and validated")
    ]
    
    for icon, stage_name, description in stages:
        st.success(f"{icon} **{stage_name}** - {description}")
    
    # Generated content preview
    st.markdown("---")
    st.subheader("📄 Generated Content")
    
    # Resume content sections
    resume_summary = workflow_state.get("resume_summary")
    resume_projects = workflow_state.get("resume_projects")
    
    if resume_summary:
        # Get the content from the GenerationResponse object
        summary_text = resume_summary.content if hasattr(resume_summary, 'content') else str(resume_summary)
        summary_metrics = get_content_metrics(summary_text, "resume")
        
        with st.expander(f"📄 Generated Resume Summary ({summary_metrics.word_count} words, {summary_metrics.estimated_pages:.1f} pages)"):
            st.text_area("Resume Summary", value=summary_text, height=150, disabled=True)
            
            # Show metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", summary_metrics.word_count)
            with col2:
                st.metric("Pages", f"{summary_metrics.estimated_pages:.1f}")
            with col3:
                st.metric("Fits 1 page", "✅" if summary_metrics.fits_one_page else "❌")
    
    if resume_projects:
        # Get the content from the GenerationResponse object
        projects_text = resume_projects.content if hasattr(resume_projects, 'content') else str(resume_projects)
        projects_metrics = get_content_metrics(projects_text, "resume")
        
        # Count projects
        import re
        project_count = len(re.findall(r'^[A-Z].*?(?=\n●)', projects_text, re.MULTILINE))
        
        with st.expander(f"🛠️ Generated Resume Projects ({project_count} projects, {projects_metrics.word_count} words)"):
            st.text_area("Resume Projects", value=projects_text, height=300, disabled=True)
            
            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Projects", project_count)
            with col2:
                st.metric("Words", projects_metrics.word_count)
            with col3:
                st.metric("Pages", f"{projects_metrics.estimated_pages:.1f}")
            with col4:
                st.metric("Fits 1 page", "✅" if projects_metrics.fits_one_page else "❌")
    
    # Cover letter content sections
    cover_letter_intro = workflow_state.get("cover_letter_intro")
    cover_letter_body = workflow_state.get("cover_letter_body")
    cover_letter_conclusion = workflow_state.get("cover_letter_conclusion")
    
    if cover_letter_intro:
        # Get the content from the GenerationResponse object
        intro_text = cover_letter_intro.content if hasattr(cover_letter_intro, 'content') else str(cover_letter_intro)
        with st.expander("💌 Generated Cover Letter Introduction"):
            st.text_area("Cover Letter Intro", value=intro_text, height=150, disabled=True)
    
    if cover_letter_body:
        # Get the content from the GenerationResponse object
        body_text = cover_letter_body.content if hasattr(cover_letter_body, 'content') else str(cover_letter_body)
        with st.expander("📝 Generated Cover Letter Body"):
            st.text_area("Cover Letter Body", value=body_text, height=200, disabled=True)
    
    if cover_letter_conclusion:
        # Get the content from the GenerationResponse object
        conclusion_text = cover_letter_conclusion.content if hasattr(cover_letter_conclusion, 'content') else str(cover_letter_conclusion)
        with st.expander("🎯 Generated Cover Letter Conclusion"):
            st.text_area("Cover Letter Conclusion", value=conclusion_text, height=150, disabled=True)
    
    # Combined cover letter view
    if cover_letter_intro and cover_letter_body and cover_letter_conclusion:
        intro_text = cover_letter_intro.content if hasattr(cover_letter_intro, 'content') else str(cover_letter_intro)
        body_text = cover_letter_body.content if hasattr(cover_letter_body, 'content') else str(cover_letter_body)
        conclusion_text = cover_letter_conclusion.content if hasattr(cover_letter_conclusion, 'content') else str(cover_letter_conclusion)
        
        combined_cover_letter = f"{intro_text}\n\n{body_text}\n\n{conclusion_text}"
        combined_metrics = get_content_metrics(combined_cover_letter, "cover_letter")
        
        with st.expander(f"💌 Complete Cover Letter ({combined_metrics.word_count} words, {combined_metrics.estimated_pages:.1f} pages)"):
            st.text_area("Full Cover Letter", value=combined_cover_letter, height=400, disabled=True)
            
            # Show combined metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Words", combined_metrics.word_count)
            with col2:
                st.metric("Pages", f"{combined_metrics.estimated_pages:.1f}")
            with col3:
                st.metric("Fits 1 page", "✅" if combined_metrics.fits_one_page else "❌")
            
            # Section breakdown
            if combined_metrics.word_count > 350:  # Cover letter limit
                st.warning("⚠️ Cover letter may exceed one page. Consider shortening content.")
            
            # Individual section metrics
            with st.expander("📊 Section Breakdown"):
                intro_words = len(intro_text.split())
                body_words = len(body_text.split())
                conclusion_words = len(conclusion_text.split())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Intro Words", intro_words, help="Target: ≤80 words")
                    if intro_words > 80:
                        st.error("Too long!")
                with col2:
                    st.metric("Body Words", body_words, help="Target: ≤160 words")
                    if body_words > 160:
                        st.error("Too long!")
                with col3:
                    st.metric("Conclusion Words", conclusion_words, help="Target: ≤60 words")
                    if conclusion_words > 60:
                        st.error("Too long!")
    
    # File downloads
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        # Resume downloads
        st.markdown("**📄 Resume Files**")
        
        resume_document = workflow_state.get("resume_document")
        
        if resume_document:
            # Get file paths from the GeneratedDocument object
            if hasattr(resume_document, 'docx_path') and resume_document.docx_path:
                try:
                    with open(resume_document.docx_path, "rb") as file:
                        st.download_button(
                            label="📄 Download Resume (DOCX)",
                            data=file.read(),
                            file_name=f"resume_{SessionManager.get_timestamp()}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                except Exception as e:
                    st.error(f"Resume DOCX download error: {str(e)}")
            
            if hasattr(resume_document, 'pdf_path') and resume_document.pdf_path:
                try:
                    with open(resume_document.pdf_path, "rb") as file:
                        st.download_button(
                            label="📄 Download Resume (PDF)",
                            data=file.read(),
                            file_name=f"resume_{SessionManager.get_timestamp()}.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"Resume PDF download error: {str(e)}")
        else:
            st.info("Resume documents not found in workflow results.")
    
    with download_col2:
        # Cover letter downloads
        st.markdown("**💌 Cover Letter Files**")
        
        cover_letter_document = workflow_state.get("cover_letter_document")
        
        if cover_letter_document:
            # Get file paths from the GeneratedDocument object
            if hasattr(cover_letter_document, 'docx_path') and cover_letter_document.docx_path:
                try:
                    with open(cover_letter_document.docx_path, "rb") as file:
                        st.download_button(
                            label="💌 Download Cover Letter (DOCX)",
                            data=file.read(),
                            file_name=f"cover_letter_{SessionManager.get_timestamp()}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                except Exception as e:
                    st.error(f"Cover Letter DOCX download error: {str(e)}")
            
            if hasattr(cover_letter_document, 'pdf_path') and cover_letter_document.pdf_path:
                try:
                    with open(cover_letter_document.pdf_path, "rb") as file:
                        st.download_button(
                            label="💌 Download Cover Letter (PDF)",
                            data=file.read(),
                            file_name=f"cover_letter_{SessionManager.get_timestamp()}.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"Cover Letter PDF download error: {str(e)}")
        else:
            st.info("Cover letter documents not found in workflow results.")
    
    # Navigation controls
    st.markdown("---")
    st.subheader("🚀 What's Next?")
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        if st.button("🔄 Run New Workflow", type="primary", use_container_width=True):
            # Clear workflow state and return to setup
            SessionManager.set_app_state_value("workflow_state", None)
            SessionManager.set_app_state_value("workflow_thread_id", None)
            SessionManager.set_app_state_value("workflow_engine", None)
            SessionManager.set_current_stage("setup")
            st.rerun()
    
    with nav_col2:
        if st.button("⚙️ Try Manual Mode", use_container_width=True):
            # Switch to manual processing mode
            SessionManager.set_current_stage("rag_processing")
            st.rerun()
    
    with nav_col3:
        if st.button("🔍 View Debug Info", use_container_width=True):
            # Show detailed workflow state for debugging
            with st.expander("🐛 Workflow Debug Information", expanded=True):
                st.json(workflow_state)
    
    # Success message
    st.success("✨ **Congratulations!** Your tailored resume and cover letter are ready. The AI workflow has successfully analyzed your background, matched it with the job requirements, and generated professional documents optimized for your target position.")


if __name__ == "__main__":
    main()