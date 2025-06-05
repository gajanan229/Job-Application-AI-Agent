"""
Main Streamlit application for the Application Factory.

This is the entry point for the Application Factory - a RAG-enhanced 
system for generating tailored resumes and cover letters.
"""

import streamlit as st
import logging
from pathlib import Path

# Import our configuration modules
from config import get_gemini_api_key, AppSettings, PathManager
from config.validators import (
    validate_master_resume, 
    validate_job_description,
    validate_streamlit_upload,
    validate_user_name,
    validate_output_path
)
from state_rag import create_initial_state, get_state_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=AppSettings.PAGE_TITLE,
    page_icon=AppSettings.PAGE_ICON,
    layout=AppSettings.LAYOUT,
    initial_sidebar_state=AppSettings.INITIAL_SIDEBAR_STATE
)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "factory_initialized" not in st.session_state:
        st.session_state.factory_initialized = False
    
    if "master_resume_content" not in st.session_state:
        st.session_state.master_resume_content = ""
    
    if "job_description_content" not in st.session_state:
        st.session_state.job_description_content = ""
    
    if "path_manager" not in st.session_state:
        st.session_state.path_manager = None
    
    if "current_state" not in st.session_state:
        st.session_state.current_state = None

def display_header():
    """Display the application header and description."""
    st.title(AppSettings.PAGE_TITLE)
    st.markdown(f"**{AppSettings.APP_DESCRIPTION}**")
    
    st.markdown("""
    Transform your Master Resume into perfectly tailored application documents using AI.
    
    **Current Status**: Phase 1 Complete - Foundation & Core Infrastructure
    """)
    
    # Display current phase progress
    with st.expander("Implementation Progress", expanded=False):
        st.markdown("""
        **Phase 1: Foundation & Core Infrastructure** *(Complete)*
        - [x] Project structure and dependencies
        - [x] Security and configuration management  
        - [x] Core state management
        - [x] Input validation system
        
        **Coming Next**:
        - Phase 2: RAG Infrastructure & Utilities
        - Phase 3: PDF Generation System
        - Phase 4: LLM Integration & Prompt Engineering
        """)

def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.title("🔧 Configuration")
    
    # API Key Section
    st.sidebar.subheader("API Key")
    api_key_method = st.sidebar.radio(
        "Choose API Key Method:",
        ["Streamlit Secrets", "Environment Variable", "Direct Input"],
        help="Select how you want to provide your Gemini API key"
    )
    
    if api_key_method == "Direct Input":
        api_key_input = st.sidebar.text_input(
            "Gemini API Key:",
            type="password",
            key="gemini_api_key_input",
            help="Enter your Google Gemini API key"
        )
    
    # Try to get the API key
    try:
        api_key = get_gemini_api_key()
        if api_key:
            st.sidebar.success("✅ API Key configured")
        else:
            st.sidebar.error("❌ No API Key found")
    except Exception as e:
        st.sidebar.error(f"❌ API Key error: {str(e)}")
    
    st.sidebar.divider()
    
    # File Upload Section
    st.sidebar.subheader("Master Resume")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Master Resume:",
        type=['txt', 'md'],
        key="master_resume_upload",
        help="Upload your comprehensive master resume (.txt or .md)"
    )
    
    if uploaded_file is not None:
        # Validate the uploaded file
        validation_result = validate_streamlit_upload(uploaded_file)
        
        if validation_result:
            try:
                content = uploaded_file.read().decode('utf-8')
                resume_validation = validate_master_resume(content)
                
                if resume_validation:
                    st.session_state.master_resume_content = content
                    st.sidebar.success("✅ Master Resume loaded")
                    
                    # Show warnings if any
                    if resume_validation.warnings:
                        with st.sidebar.expander("⚠️ Resume Analysis", expanded=False):
                            for warning in resume_validation.warnings:
                                st.warning(warning)
                else:
                    st.sidebar.error(f"❌ {resume_validation.message}")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error reading file: {str(e)}")
        else:
            st.sidebar.error(f"❌ {validation_result.message}")
    
    st.sidebar.divider()
    
    # Output Configuration
    st.sidebar.subheader("Output Settings")
    output_path = st.sidebar.text_input(
        "Output Directory:",
        value=AppSettings.DEFAULT_OUTPUT_DIR,
        key="output_base_path_input",
        help="Directory where generated files will be saved"
    )
    
    user_name = st.sidebar.text_input(
        "Your Name:",
        key="user_name_input",
        help="Name to use in generated file names"
    )
    
    # Initialize PathManager if output path is valid
    if output_path:
        path_validation = validate_output_path(output_path)
        if path_validation:
            try:
                st.session_state.path_manager = PathManager(output_path)
                st.sidebar.success("✅ Output path configured")
            except Exception as e:
                st.sidebar.error(f"❌ Path error: {str(e)}")
        else:
            st.sidebar.error(f"❌ {path_validation.message}")

def main_interface():
    """Display the main application interface."""
    
    # Check if basic setup is complete
    has_api_key = bool(get_gemini_api_key())
    has_resume = bool(st.session_state.master_resume_content)
    has_path_manager = st.session_state.path_manager is not None
    
    if not has_api_key:
        st.warning("⚠️ Please configure your Gemini API key in the sidebar to continue.")
        return
    
    if not has_resume:
        st.info("Please upload your Master Resume in the sidebar to begin.")
        return
    
    if not has_path_manager:
        st.warning("⚠️ Please configure a valid output directory in the sidebar.")
        return
    
    # Main interface sections
    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        height=200,
        key="job_description_input",
        help="Copy and paste the complete job posting you want to apply for"
    )
    
    if job_description:
        # Validate job description
        jd_validation = validate_job_description(job_description)
        
        if jd_validation:
            st.session_state.job_description_content = job_description
            st.success("✅ Job description loaded and validated")
            
            # Show warnings if any
            if jd_validation.warnings:
                with st.expander("⚠️ Job Description Analysis", expanded=False):
                    for warning in jd_validation.warnings:
                        st.warning(warning)
            
            # Show what the path manager would create
            if st.session_state.path_manager:
                company, position = st.session_state.path_manager.extract_company_and_position(job_description)
                st.info(f"Will create folder: `{company}_{position}`")
        else:
            st.error(f"❌ {jd_validation.message}")
    
    # Show current status
    if has_api_key and has_resume and has_path_manager and job_description:
        st.divider()
        st.subheader("Ready to Generate")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Initialize Application Factory", type="primary"):
                # Create initial state (Phase 1 demonstration)
                try:
                    initial_state = create_initial_state(
                        master_resume_path="uploaded_resume.txt",
                        job_description_content=st.session_state.job_description_content,
                        output_base_path=st.session_state.path_manager.base_output_path,
                        resume_header=st.session_state.get("user_name_input", ""),
                        cover_letter_header=""
                    )
                    
                    st.session_state.current_state = initial_state
                    st.session_state.factory_initialized = True
                    st.success("✅ Application Factory initialized successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Initialization error: {str(e)}")
        
        with col2:
            if st.session_state.factory_initialized:
                st.success("✅ Factory Ready")
            else:
                st.info("⏳ Awaiting initialization")
    
    # Show state information if initialized
    if st.session_state.current_state:
        st.divider()
        st.subheader("Current State")
        
        state_summary = get_state_summary(st.session_state.current_state)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Stage", state_summary["stage"])
            st.metric("Resume Sections", f"{state_summary['resume_sections_count']}/5")
        
        with col2:
            st.metric("Master Resume", "✅" if state_summary["master_resume_loaded"] else "❌")
            st.metric("Job Description", "✅" if state_summary["job_description_loaded"] else "❌")
        
        with col3:
            st.metric("Vector Store", "✅" if state_summary["vector_store_ready"] else "❌")
            st.metric("Has Error", "❌" if state_summary["has_error"] else "✅")
        
        # Show detailed state in expander
        with st.expander("Detailed State Information", expanded=False):
            st.json(state_summary)

def display_footer():
    """Display footer information."""
    st.divider()
    st.markdown("""
    ---
    **Application Factory v1.0.0** | Phase 1: Foundation & Core Infrastructure Complete
    
    🔗 Part of the larger AI-powered job application system
    """)

def main():
    """Main application function."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Display header
        display_header()
        
        # Setup sidebar
        setup_sidebar()
        
        # Main interface
        main_interface()
        
        # Footer
        display_footer()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 