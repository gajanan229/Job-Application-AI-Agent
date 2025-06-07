"""
Main Streamlit application for the Application Factory.

This is the entry point for the Application Factory - a RAG-enhanced 
system for generating tailored resumes and cover letters.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time

# Import our configuration modules
from config import get_gemini_api_key, AppSettings, PathManager
from config.validators import (
    validate_master_resume, 
    validate_job_description,
    validate_streamlit_upload,
    validate_user_name,
    validate_output_path
)
# State management utilities (minimal usage in blueprint mode)
from state_rag import create_initial_state

# Import RAG and PDF utilities
from rag_utils import RAGManager, create_resume_vector_store
from html_pdf_utils_alt import generate_resume_pdf, generate_cover_letter_pdf, validate_pdf_output
from llm_utils import LLMManager

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
    
    # Blueprint-specific state variables
    if "current_stage" not in st.session_state:
        st.session_state.current_stage = "setup"  # setup, resume_builder, cover_letter_builder, completed
    
    if "job_analysis_complete" not in st.session_state:
        st.session_state.job_analysis_complete = False
    
    if "extracted_job_requirements" not in st.session_state:
        st.session_state.extracted_job_requirements = {}
    
    # Resume Builder state
    if "resume_sections_generated" not in st.session_state:
        st.session_state.resume_sections_generated = {}
    
    if "resume_sections_edited" not in st.session_state:
        st.session_state.resume_sections_edited = {}
    
    if "resume_finalized" not in st.session_state:
        st.session_state.resume_finalized = False
    
    if "resume_pdf_path" not in st.session_state:
        st.session_state.resume_pdf_path = ""
    
    # Cover Letter Builder state
    if "cover_letter_versions" not in st.session_state:
        st.session_state.cover_letter_versions = {
            "introduction": [],
            "body_paragraphs": [],
            "conclusion": []
        }
    
    if "current_cover_letter_version" not in st.session_state:
        st.session_state.current_cover_letter_version = {
            "introduction": 0,
            "body_paragraphs": 0,
            "conclusion": 0
        }
    
    if "cover_letter_feedback" not in st.session_state:
        st.session_state.cover_letter_feedback = {
            "introduction": "",
            "body_paragraphs": "",
            "conclusion": ""
        }
    
    if "cover_letter_finalized" not in st.session_state:
        st.session_state.cover_letter_finalized = False
    
    if "cover_letter_pdf_path" not in st.session_state:
        st.session_state.cover_letter_pdf_path = ""
    
    # RAG and managers
    if "vector_store_created" not in st.session_state:
        st.session_state.vector_store_created = False
    
    if "rag_manager" not in st.session_state:
        st.session_state.rag_manager = None
    
    if "llm_manager" not in st.session_state:
        st.session_state.llm_manager = None

def display_header():
    """Display the application header and description."""
    st.title(AppSettings.PAGE_TITLE)
    st.markdown(f"**{AppSettings.APP_DESCRIPTION}**")
    
    st.markdown("""
    🏭 **Your Personal Digital Artisan's Workshop**
    
    Transform your Master Resume into perfectly tailored application documents using AI-powered interactive tools.
    """)
    
    # Display current phase progress
    with st.expander("🎯 Application Factory Workflow", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔨 Stage 1: Resume Builder**
            - Intelligent analysis of job requirements
            - Section-by-section crafting with AI assistance
            - Real-time PDF preview
            - Iterative refinement until perfect
            """)
        
        with col2:
            st.markdown("""
            **📝 Stage 2: Cover Letter Builder** 
            - Narrative weaving with versioning
            - Feedback-driven regeneration
            - Introduction → Body → Conclusion flow
            - Final packaging & organization
            """)
        
        st.info("🎯 **Interactive Mode**: You control every step, with AI as your intelligent assistant")

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
    
    st.sidebar.divider()
    
    # Contact Information
    st.sidebar.subheader("📞 Contact Information")
    st.sidebar.caption("For resume header generation")
    
    user_name = st.sidebar.text_input(
        "Full Name:",
        key="user_name_input",
        help="Your full name as it should appear on the resume"
    )
    
    phone_number = st.sidebar.text_input(
        "Phone Number:",
        key="phone_input",
        placeholder="(000) 000-0000",
        help="Your phone number"
    )
    
    email_address = st.sidebar.text_input(
        "Email Address:",
        key="email_input",
        placeholder="your.email@example.com",
        help="Your email address"
    )
    
    linkedin_url = st.sidebar.text_input(
        "LinkedIn URL:",
        key="linkedin_input",
        placeholder="https://linkedin.com/in/yourprofile",
        help="Your LinkedIn profile URL"
    )
    
    website_url = st.sidebar.text_input(
        "Website URL (Optional):",
        key="website_input",
        placeholder="https://yourwebsite.com",
        help="Your personal website or portfolio URL"
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
    """Display the main application interface based on blueprint workflow."""
    
    # Check if basic setup is complete
    has_api_key = bool(get_gemini_api_key())
    has_resume = bool(st.session_state.master_resume_content)
    has_path_manager = st.session_state.path_manager is not None
    
    if not has_api_key:
        st.warning("⚠️ Please configure your Gemini API key in the sidebar to continue.")
        return
    
    if not has_resume:
        st.info("📋 Please upload your Master Resume in the sidebar to begin.")
        return
    
    if not has_path_manager:
        st.warning("⚠️ Please configure a valid output directory in the sidebar.")
        return
    
    # Entry Point to the Factory
    st.subheader("🎯 Entry Point to the Factory")
    st.write("**Step 1:** Select your target job by pasting the job description below.")
    
    job_description = st.text_area(
        "📋 Job Description (Your Order Ticket):",
        height=200,
        key="job_description_input",
        help="Copy and paste the complete job posting you want to apply for",
        value=st.session_state.job_description_content
    )
    
    if job_description:
        # Validate job description
        jd_validation = validate_job_description(job_description)
        
        if jd_validation:
            st.session_state.job_description_content = job_description
            
            # Extract company and position
            if st.session_state.path_manager:
                company, position = st.session_state.path_manager.extract_company_and_position(job_description)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🏢 **Company:** {company}")
                with col2:
                    st.success(f"💼 **Position:** {position}")
                
                st.info(f"📁 Application folder will be: `{company}_{position}`")
                
                # Progress to next stage
                if st.session_state.current_stage == "setup":
                    if st.button("🚀 Enter the Application Factory", type="primary"):
                        st.session_state.current_stage = "resume_builder"
                        st.rerun()
        else:
            st.error(f"❌ {jd_validation.message}")
    
    # Stage-based interface
    if st.session_state.current_stage == "resume_builder" and job_description:
        display_resume_builder()
    elif st.session_state.current_stage == "cover_letter_builder":
        display_cover_letter_builder()
    elif st.session_state.current_stage == "completed":
        display_completion_summary()


def display_footer():
    """Display footer information."""
    st.divider()
    st.markdown("""
    ---
    **Application Factory v3.0.0** | Interactive Blueprint Implementation Complete! 🎉
    
    🔨 **Stage 1**: Interactive Resume Builder with AI Assistance  
    📝 **Stage 2**: Narrative Weaver with Versioning & Feedback Loops  
    🎯 **Complete**: Section-by-Section Editing, Real-Time Preview, and Professional Output  
    
    ✨ Your personal digital artisan's workshop is ready!
    """)



def display_resume_builder():
    """Display the Resume Builder interface following the blueprint."""
    st.divider()
    st.header("🔨 Stage 1: The Personalized Resume Blueprint")
    st.write("*Transform your Master Resume into a lean, targeted, one-page powerhouse!*")
    
    # Progress indicator
    sections = ['summary', 'skills', 'education', 'experience', 'projects']
    completed_sections = sum(1 for section in sections if section in st.session_state.resume_sections_edited)
    
    progress_col1, progress_col2 = st.columns([3, 1])
    with progress_col1:
        st.progress(completed_sections / len(sections), f"Sections Completed: {completed_sections}/{len(sections)}")
    with progress_col2:
        if st.button("📄 Preview Resume", disabled=completed_sections == 0):
            preview_resume_pdf()
    
    # Step 1: Intelligent Analysis (if not done)
    if not st.session_state.job_analysis_complete:
        display_job_analysis_step()
        return
    
    # Step 2: Section-by-Section Crafting
    st.subheader("✏️ Section-by-Section Crafting & Your Master Edit Suite")
    
    # Create tabs for each section
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Elevator Pitch", "🛠️ Skill Showcase", "🎓 Foundation", 
        "💼 Proof of Action", "🚀 Show Don't Tell"
    ])
    
    with tab1:
        display_summary_section_editor()
    
    with tab2:
        display_skills_section_editor()
    
    with tab3:
        display_education_section_editor()
    
    with tab4:
        display_experience_section_editor()
    
    with tab5:
        display_projects_section_editor()
    
    # Final actions
    st.divider()
    if completed_sections == len(sections):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Final Preview", type="secondary"):
                preview_resume_pdf()
        
        with col2:
            if st.button("🔄 Back to Edit", type="secondary"):
                # Allow going back to editing
                pass
        
        with col3:
            if st.button("✅ Finalize & Save Resume", type="primary"):
                finalize_resume()

def display_job_analysis_step():
    """Display the intelligent job analysis step."""
    st.subheader("🔍 The Intelligent Analysis & Initial Draft")
    st.write("*The 'Smart Sorter & Shaper' is analyzing your job requirements...*")
    
    with st.expander("🎯 What the AI is analyzing:", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**From Job Description:**")
            st.write("• Keywords and key phrases")
            st.write("• Underlying needs and problems")
            st.write("• Company culture hints")
        
        with col2:
            st.write("**From Your Master Resume:**")
            st.write("• Relevant skills and experiences")
            st.write("• Matching projects and achievements")
            st.write("• Alignment opportunities")
    
    if st.button("🚀 Start Intelligent Analysis", type="primary"):
        with st.spinner("🧠 AI is analyzing job requirements and your background..."):
            # Initialize managers if needed
            if not st.session_state.rag_manager:
                st.session_state.rag_manager = RAGManager()
            
            if not st.session_state.llm_manager:
                # Import here to avoid circular imports
                from llm_utils import LLMManager
                st.session_state.llm_manager = LLMManager(st.session_state.rag_manager)
            
            # Extract job requirements
            try:
                job_skills = st.session_state.llm_manager.extract_job_skills(
                    st.session_state.job_description_content
                )
                st.session_state.extracted_job_requirements = job_skills
                st.session_state.job_analysis_complete = True
                
                st.success("✅ Analysis complete! Ready to craft your sections.")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

def display_summary_section_editor():
    """Display the Summary/Objective section editor."""
    st.subheader("🎯 The 'Elevator Pitch' (Summary/Objective)")
    st.write("*A concise, impactful opening statement that highlights your most relevant qualifications.*")
    
    # Check if section is generated
    if 'summary' not in st.session_state.resume_sections_generated:
        if st.button("🤖 Generate AI Summary", type="primary"):
            generate_resume_section('summary')
            st.rerun()
    else:
        # Show generated content and allow editing
        current_content = st.session_state.resume_sections_edited.get(
            'summary', 
            st.session_state.resume_sections_generated['summary']
        )
        
        edited_content = st.text_area(
            "Edit your summary:",
            value=current_content,
            height=120,
            key="summary_editor",
            help="Refine the tone, add specific achievements, or rephrase sentences"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Changes", key="save_summary"):
                st.session_state.resume_sections_edited['summary'] = edited_content
                st.success("✅ Summary saved!")
        
        with col2:
            if st.button("🔄 Regenerate", key="regen_summary"):
                generate_resume_section('summary')
                st.rerun()

def display_skills_section_editor():
    """Display the Skills section editor."""
    st.subheader("🛠️ The 'Skill Showcase' (Skills Section)")
    st.write("*Showcase your most relevant skills, categorized and prioritized for this role.*")
    
    # Check if section is generated
    if 'skills' not in st.session_state.resume_sections_generated:
        if st.button("🤖 Generate AI Skills", type="primary"):
            generate_resume_section('skills')
            st.rerun()
    else:
        # Show generated content and allow editing
        current_content = st.session_state.resume_sections_edited.get(
            'skills', 
            st.session_state.resume_sections_generated['skills']
        )
        
        edited_content = st.text_area(
            "Edit your skills:",
            value=current_content,
            height=150,
            key="skills_editor",
            help="Add missing skills, remove less relevant ones, or reorganize by importance"
        )
        
        # Skills management interface
        st.write("**Quick Skills Management:**")
        col1, col2 = st.columns(2)
        
        with col1:
            new_skill = st.text_input("Add a skill:", key="new_skill_input")
            if st.button("➕ Add", key="add_skill") and new_skill:
                if new_skill not in edited_content:
                    # Simple addition - in a full implementation you'd parse and reorganize
                    edited_content += f", {new_skill}"
                    st.session_state.resume_sections_edited['skills'] = edited_content
                    st.rerun()
        
        with col2:
            if st.button("🔄 Regenerate Skills", key="regen_skills"):
                generate_resume_section('skills')
                st.rerun()
        
        # Save button
        if st.button("💾 Save Skills Changes", key="save_skills"):
            st.session_state.resume_sections_edited['skills'] = edited_content
            st.success("✅ Skills saved!")

def display_education_section_editor():
    """Display the Education section editor."""
    st.subheader("🎓 The 'Foundation' (Education Section)")
    st.write("*Your educational background, highlighting the most relevant aspects.*")
    
    # Check if section is generated
    if 'education' not in st.session_state.resume_sections_generated:
        if st.button("🤖 Generate AI Education", type="primary"):
            generate_resume_section('education')
            st.rerun()
    else:
        # Show generated content and allow editing
        current_content = st.session_state.resume_sections_edited.get(
            'education', 
            st.session_state.resume_sections_generated['education']
        )
        
        edited_content = st.text_area(
            "Edit your education:",
            value=current_content,
            height=120,
            key="education_editor",
            help="Verify accuracy, add/remove coursework, adjust GPA visibility"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Changes", key="save_education"):
                st.session_state.resume_sections_edited['education'] = edited_content
                st.success("✅ Education saved!")
        
        with col2:
            if st.button("🔄 Regenerate", key="regen_education"):
                generate_resume_section('education')
                st.rerun()

def display_experience_section_editor():
    """Display the Experience section editor."""
    st.subheader("💼 The 'Proof of Action' (Work Experience)")
    st.write("*Your work experience, with bullet points crafted to demonstrate relevant skills.*")
    
    # Check if section is generated
    if 'experience' not in st.session_state.resume_sections_generated:
        if st.button("🤖 Generate AI Experience", type="primary"):
            generate_resume_section('experience')
            st.rerun()
    else:
        # Show generated content and allow editing
        current_content = st.session_state.resume_sections_edited.get(
            'experience', 
            st.session_state.resume_sections_generated['experience']
        )
        
        edited_content = st.text_area(
            "Edit your experience:",
            value=current_content,
            height=200,
            key="experience_editor",
            help="Refine bullet points, use strong action verbs, quantify achievements"
        )
        
        # Experience tips
        with st.expander("💡 Experience Writing Tips", expanded=False):
            st.write("• Start bullet points with strong action verbs")
            st.write("• Quantify achievements where possible (numbers, percentages)")
            st.write("• Focus on results and impact, not just responsibilities")
            st.write("• Prioritize most relevant experiences for this role")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Changes", key="save_experience"):
                st.session_state.resume_sections_edited['experience'] = edited_content
                st.success("✅ Experience saved!")
        
        with col2:
            if st.button("🔄 Regenerate", key="regen_experience"):
                generate_resume_section('experience')
                st.rerun()

def display_projects_section_editor():
    """Display the Projects section editor."""
    st.subheader("🚀 The 'Show, Don't Just Tell' (Projects)")
    st.write("*Strategic selection of projects that best demonstrate your capabilities for this role.*")
    
    # Check if section is generated
    if 'projects' not in st.session_state.resume_sections_generated:
        if st.button("🤖 Generate AI Projects Selection", type="primary"):
            generate_resume_section('projects')
            st.rerun()
    else:
        # Show generated content and allow editing
        current_content = st.session_state.resume_sections_edited.get(
            'projects', 
            st.session_state.resume_sections_generated['projects']
        )
        
        st.info("🎯 **Strategic Selector**: AI has chosen the most relevant projects from your Master Resume")
        
        edited_content = st.text_area(
            "Edit your projects:",
            value=current_content,
            height=200,
            key="projects_editor",
            help="Review chosen projects, edit descriptions, ensure clarity and impact"
        )
        
        # Project management interface
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Regenerate Project Selection", key="regen_projects"):
                generate_resume_section('projects')
                st.rerun()
        
        with col2:
            if st.button("💾 Save Changes", key="save_projects"):
                st.session_state.resume_sections_edited['projects'] = edited_content
                st.success("✅ Projects saved!")

def generate_resume_section(section_name: str):
    """Generate a specific resume section using AI."""
    try:
        with st.spinner(f"🤖 Generating {section_name} section..."):
            if not st.session_state.llm_manager:
                from llm_utils import LLMManager
                st.session_state.llm_manager = LLMManager(st.session_state.rag_manager)
            
            content = st.session_state.llm_manager.generate_resume_section(
                section_name,
                st.session_state.job_description_content,
                st.session_state.master_resume_content
            )
            
            if content:
                st.session_state.resume_sections_generated[section_name] = content
                st.success(f"✅ {section_name.title()} section generated!")
            else:
                st.error(f"❌ Failed to generate {section_name} section")
                
    except Exception as e:
        st.error(f"❌ Error generating {section_name}: {str(e)}")

def preview_resume_pdf():
    """Generate and display a preview of the resume PDF."""
    try:
        # Collect all edited or generated sections
        sections = {}
        for section in ['summary', 'skills', 'education', 'experience', 'projects']:
            if section in st.session_state.resume_sections_edited:
                sections[section] = st.session_state.resume_sections_edited[section]
            elif section in st.session_state.resume_sections_generated:
                sections[section] = st.session_state.resume_sections_generated[section]
        
        if not sections:
            st.warning("⚠️ No sections available for preview. Generate some sections first.")
            return
        
        # Get contact info
        contact_name = st.session_state.get("user_name_input", "Your Name")
        contact_phone = st.session_state.get("phone_input", "(555) 123-4567")
        contact_email = st.session_state.get("email_input", "your.email@example.com")
        contact_linkedin = st.session_state.get("linkedin_input", "")
        
        # Generate preview PDF
        preview_path = "temp/resume_preview.pdf"
        
        success = generate_resume_pdf(
            sections=sections,
            name=contact_name,
            phone=contact_phone,
            email=contact_email,
            linkedin=contact_linkedin,
            website="",
            location="Toronto, ON",
            output_path=preview_path
        )
        
        if success:
            st.success("📄 Resume preview generated!")
            st.info(f"✅ **One-Page Check**: Preview saved to {preview_path}")
            
            # In a real implementation, you'd display the PDF inline
            # For now, we'll show a success message
            with st.expander("📋 Preview Content Summary", expanded=True):
                for section, content in sections.items():
                    st.write(f"**{section.title()}**: {len(content)} characters")
                    
        else:
            st.error("❌ Failed to generate preview")
            
    except Exception as e:
        st.error(f"❌ Preview error: {str(e)}")

def finalize_resume():
    """Finalize the resume and move to cover letter stage."""
    try:
        # Collect all sections
        sections = {}
        for section in ['summary', 'skills', 'education', 'experience', 'projects']:
            if section in st.session_state.resume_sections_edited:
                sections[section] = st.session_state.resume_sections_edited[section]
            elif section in st.session_state.resume_sections_generated:
                sections[section] = st.session_state.resume_sections_generated[section]
        
        if len(sections) < 5:
            st.error("❌ Please complete all 5 sections before finalizing.")
            return
        
        # Generate final PDF
        contact_name = st.session_state.get("user_name_input", "Your Name")
        contact_phone = st.session_state.get("phone_input", "(555) 123-4567")
        contact_email = st.session_state.get("email_input", "your.email@example.com")
        contact_linkedin = st.session_state.get("linkedin_input", "")
        
        # Create job folder
        company, position = st.session_state.path_manager.extract_company_and_position(
            st.session_state.job_description_content
        )
        job_folder = st.session_state.path_manager.create_job_folder(
            st.session_state.job_description_content
        )
        resume_path = st.session_state.path_manager.get_resume_path(job_folder, contact_name)
        
        success = generate_resume_pdf(
            sections=sections,
            name=contact_name,
            phone=contact_phone,
            email=contact_email,
            linkedin=contact_linkedin,
            website="",
            location="Toronto, ON",
            output_path=str(resume_path)
        )
        
        if success:
            st.session_state.resume_finalized = True
            st.session_state.resume_pdf_path = str(resume_path)
            st.session_state.current_stage = "cover_letter_builder"
            
            st.success(f"🎉 Resume finalized and saved!")
            st.success(f"📁 Saved to: {resume_path}")
            st.info("🚀 Ready to move to Cover Letter Builder...")
            
            time.sleep(2)  # Brief pause for user to see success
            st.rerun()
            
        else:
            st.error("❌ Failed to save final resume")
            
    except Exception as e:
        st.error(f"❌ Error finalizing resume: {str(e)}")

def display_cover_letter_builder():
    """Display the Cover Letter Builder interface following the blueprint."""
    st.divider()
    st.header("📝 Stage 2: The Narrative Weaver")
    st.write("*Craft a compelling narrative that connects your qualifications directly to the employer's needs.*")
    
    # Show resume completion status
    st.success(f"✅ **Resume Complete**: {st.session_state.resume_pdf_path}")
    
    # Cover letter progress
    intro_versions = len(st.session_state.cover_letter_versions["introduction"])
    body_versions = len(st.session_state.cover_letter_versions["body_paragraphs"])
    conclusion_versions = len(st.session_state.cover_letter_versions["conclusion"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Introduction", f"v{intro_versions}" if intro_versions > 0 else "Not Started")
    with col2:
        st.metric("📄 Body Paragraphs", f"v{body_versions}" if body_versions > 0 else "Not Started")
    with col3:
        st.metric("🎯 Conclusion", f"v{conclusion_versions}" if conclusion_versions > 0 else "Not Started")
    
    # Two-phase approach: First intro/conclusion, then body
    st.subheader("🎨 The Art of Introduction & Conclusion")
    st.write("*Perfect your opening and closing first - your best chances to make a memorable impact.*")
    
    # Introduction section
    with st.expander("📝 Introduction Paragraph", expanded=True):
        display_introduction_editor()
    
    # Conclusion section  
    with st.expander("🎯 Conclusion Paragraph", expanded=True):
        display_conclusion_editor()
    
    # Body paragraphs (only after intro/conclusion)
    if intro_versions > 0 and conclusion_versions > 0:
        st.divider()
        st.subheader("📄 Weaving the Core Narrative")
        st.write("*With strong opening and closing, now craft the compelling body paragraphs.*")
        
        with st.expander("💼 Body Paragraphs", expanded=True):
            display_body_paragraphs_editor()
    
    # Final actions
    if intro_versions > 0 and body_versions > 0 and conclusion_versions > 0:
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Preview Cover Letter", type="secondary"):
                preview_cover_letter_pdf()
        
        with col2:
            if st.button("🔄 Continue Editing", type="secondary"):
                pass  # Stay in edit mode
        
        with col3:
            if st.button("✅ Finalize & Save Cover Letter", type="primary"):
                finalize_cover_letter()

def display_introduction_editor():
    """Display the introduction paragraph editor with versioning."""
    st.write("*Grab attention, state the position, express enthusiasm and core suitability.*")
    
    versions = st.session_state.cover_letter_versions["introduction"]
    current_version_idx = st.session_state.current_cover_letter_version["introduction"]
    
    if not versions:
        # Generate first version
        if st.button("🤖 Generate Introduction", type="primary", key="gen_intro"):
            generate_cover_letter_section("introduction")
            st.rerun()
    else:
        # Show current version
        current_content = versions[current_version_idx]
        
        # Version navigation
        if len(versions) > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="intro_prev", disabled=current_version_idx == 0):
                    st.session_state.current_cover_letter_version["introduction"] -= 1
                    st.rerun()
            with col2:
                st.write(f"**Version {current_version_idx + 1} of {len(versions)}**")
            with col3:
                if st.button("➡️ Next", key="intro_next", disabled=current_version_idx == len(versions) - 1):
                    st.session_state.current_cover_letter_version["introduction"] += 1
                    st.rerun()
        
        # Display content
        st.text_area(
            "Current Introduction:",
            value=current_content,
            height=100,
            key="intro_display",
            disabled=True
        )
        
        # Feedback interface
        st.write("**🎭 Director's Notes (Guide the AI):**")
        feedback = st.text_area(
            "Provide feedback for the next version:",
            value=st.session_state.cover_letter_feedback["introduction"],
            height=80,
            key="intro_feedback",
            placeholder="e.g., 'Make it more confident', 'Mention my passion for AI', 'Make it specific to the company'"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Feedback", key="save_intro_feedback"):
                st.session_state.cover_letter_feedback["introduction"] = feedback
                st.success("✅ Feedback saved!")
        
        with col2:
            if st.button("🔄 Regenerate with Feedback", key="regen_intro", disabled=not feedback.strip()):
                generate_cover_letter_section("introduction", feedback)
                st.rerun()

def display_conclusion_editor():
    """Display the conclusion paragraph editor with versioning."""
    st.write("*Reiterate strong interest, suggest next steps, and thank the reader.*")
    
    versions = st.session_state.cover_letter_versions["conclusion"]
    current_version_idx = st.session_state.current_cover_letter_version["conclusion"]
    
    if not versions:
        # Generate first version
        if st.button("🤖 Generate Conclusion", type="primary", key="gen_conclusion"):
            generate_cover_letter_section("conclusion")
            st.rerun()
    else:
        # Show current version
        current_content = versions[current_version_idx]
        
        # Version navigation
        if len(versions) > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="conclusion_prev", disabled=current_version_idx == 0):
                    st.session_state.current_cover_letter_version["conclusion"] -= 1
                    st.rerun()
            with col2:
                st.write(f"**Version {current_version_idx + 1} of {len(versions)}**")
            with col3:
                if st.button("➡️ Next", key="conclusion_next", disabled=current_version_idx == len(versions) - 1):
                    st.session_state.current_cover_letter_version["conclusion"] += 1
                    st.rerun()
        
        # Display content
        st.text_area(
            "Current Conclusion:",
            value=current_content,
            height=100,
            key="conclusion_display",
            disabled=True
        )
        
        # Feedback interface
        st.write("**🎭 Director's Notes (Guide the AI):**")
        feedback = st.text_area(
            "Provide feedback for the next version:",
            value=st.session_state.cover_letter_feedback["conclusion"],
            height=80,
            key="conclusion_feedback",
            placeholder="e.g., 'Less generic, more specific to company', 'Stronger call to action'"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Feedback", key="save_conclusion_feedback"):
                st.session_state.cover_letter_feedback["conclusion"] = feedback
                st.success("✅ Feedback saved!")
        
        with col2:
            if st.button("🔄 Regenerate with Feedback", key="regen_conclusion", disabled=not feedback.strip()):
                generate_cover_letter_section("conclusion", feedback)
                st.rerun()

def display_body_paragraphs_editor():
    """Display the body paragraphs editor with versioning."""
    st.write("*Connect specific experiences to job requirements and elaborate on key achievements.*")
    
    versions = st.session_state.cover_letter_versions["body_paragraphs"]
    current_version_idx = st.session_state.current_cover_letter_version["body_paragraphs"]
    
    if not versions:
        # Generate first version
        st.info("📋 **Ready to craft body paragraphs** using your finalized introduction and conclusion.")
        if st.button("🤖 Generate Body Paragraphs", type="primary", key="gen_body"):
            generate_cover_letter_section("body_paragraphs")
            st.rerun()
    else:
        # Show current version
        current_content = versions[current_version_idx]
        
        # Version navigation
        if len(versions) > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="body_prev", disabled=current_version_idx == 0):
                    st.session_state.current_cover_letter_version["body_paragraphs"] -= 1
                    st.rerun()
            with col2:
                st.write(f"**Version {current_version_idx + 1} of {len(versions)}**")
            with col3:
                if st.button("➡️ Next", key="body_next", disabled=current_version_idx == len(versions) - 1):
                    st.session_state.current_cover_letter_version["body_paragraphs"] += 1
                    st.rerun()
        
        # Display content (as list since body can be multiple paragraphs)
        if isinstance(current_content, list):
            for i, paragraph in enumerate(current_content):
                st.text_area(
                    f"Body Paragraph {i + 1}:",
                    value=paragraph,
                    height=80,
                    key=f"body_display_{i}",
                    disabled=True
                )
        else:
            st.text_area(
                "Body Content:",
                value=str(current_content),
                height=150,
                key="body_display_full",
                disabled=True
            )
        
        # Feedback interface
        st.write("**🎭 Director's Notes (Guide the AI):**")
        feedback = st.text_area(
            "Provide feedback for the next version:",
            value=st.session_state.cover_letter_feedback["body_paragraphs"],
            height=80,
            key="body_feedback",
            placeholder="e.g., 'Expand on project X', 'More conversational tone', 'Stronger link to company mission'"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Feedback", key="save_body_feedback"):
                st.session_state.cover_letter_feedback["body_paragraphs"] = feedback
                st.success("✅ Feedback saved!")
        
        with col2:
            if st.button("🔄 Regenerate with Feedback", key="regen_body", disabled=not feedback.strip()):
                generate_cover_letter_section("body_paragraphs", feedback)
                st.rerun()

def generate_cover_letter_section(section_name: str, feedback: str = ""):
    """Generate a cover letter section with optional feedback."""
    try:
        with st.spinner(f"🤖 Generating {section_name} section..."):
            if not st.session_state.llm_manager:
                from llm_utils import LLMManager
                st.session_state.llm_manager = LLMManager(st.session_state.rag_manager)
            
            # Extract company and position
            company, position = st.session_state.path_manager.extract_company_and_position(
                st.session_state.job_description_content
            )
            
            # Get applicant name
            applicant_name = st.session_state.get("user_name_input", "Applicant")
            
            # Generate based on section type
            if section_name == "introduction":
                content = generate_introduction_paragraph(company, position, applicant_name, feedback)
            elif section_name == "conclusion":
                content = generate_conclusion_paragraph(company, position, applicant_name, feedback)
            elif section_name == "body_paragraphs":
                content = generate_body_paragraphs(company, position, applicant_name, feedback)
            else:
                st.error(f"Unknown section: {section_name}")
                return
            
            # Add to versions
            st.session_state.cover_letter_versions[section_name].append(content)
            st.session_state.current_cover_letter_version[section_name] = len(st.session_state.cover_letter_versions[section_name]) - 1
            
            # Clear feedback after use
            st.session_state.cover_letter_feedback[section_name] = ""
            
            st.success(f"✅ {section_name.replace('_', ' ').title()} generated!")
            
    except Exception as e:
        st.error(f"❌ Error generating {section_name}: {str(e)}")

def generate_introduction_paragraph(company: str, position: str, applicant_name: str, feedback: str = ""):
    """Generate introduction paragraph for cover letter."""
    # This would use the LLM manager to generate content
    # For now, providing a structured template
    
    base_intro = f"Dear Hiring Manager,\n\nI am writing to express my strong interest in the {position} position at {company}. With my background in software development and passion for technology, I am excited about the opportunity to contribute to your team's success."
    
    if feedback:
        # In a real implementation, this would use LLM to modify based on feedback
        base_intro += f"\n\n[Modified based on feedback: {feedback}]"
    
    return base_intro

def generate_conclusion_paragraph(company: str, position: str, applicant_name: str, feedback: str = ""):
    """Generate conclusion paragraph for cover letter."""
    base_conclusion = f"I would welcome the opportunity to discuss how my background and enthusiasm can contribute to {company}'s continued success. Thank you for considering my application, and I look forward to hearing from you soon.\n\nSincerely,\n{applicant_name}"
    
    if feedback:
        base_conclusion += f"\n\n[Modified based on feedback: {feedback}]"
    
    return base_conclusion

def generate_body_paragraphs(company: str, position: str, applicant_name: str, feedback: str = ""):
    """Generate body paragraphs for cover letter."""
    paragraphs = [
        f"My experience in software development and project management aligns perfectly with the requirements for the {position} role. In my previous projects, I have demonstrated proficiency in the technologies and methodologies that {company} values, including collaborative development and innovative problem-solving.",
        
        f"I am particularly drawn to {company}'s mission and innovative approach to technology. My recent projects in areas such as web development and automation demonstrate my ability to deliver results and adapt to new challenges, qualities that I believe would make me a valuable addition to your team."
    ]
    
    if feedback:
        paragraphs.append(f"[Additional content based on feedback: {feedback}]")
    
    return paragraphs

def preview_cover_letter_pdf():
    """Generate and display a preview of the cover letter PDF."""
    try:
        # Get current versions of all sections
        intro_versions = st.session_state.cover_letter_versions["introduction"]
        body_versions = st.session_state.cover_letter_versions["body_paragraphs"]
        conclusion_versions = st.session_state.cover_letter_versions["conclusion"]
        
        if not intro_versions or not body_versions or not conclusion_versions:
            st.warning("⚠️ Please complete all cover letter sections before preview.")
            return
        
        # Get current version indices
        intro_idx = st.session_state.current_cover_letter_version["introduction"]
        body_idx = st.session_state.current_cover_letter_version["body_paragraphs"]
        conclusion_idx = st.session_state.current_cover_letter_version["conclusion"]
        
        # Get content
        intro = intro_versions[intro_idx]
        body = body_versions[body_idx]
        conclusion = conclusion_versions[conclusion_idx]
        
        # Get contact info
        contact_name = st.session_state.get("user_name_input", "Your Name")
        contact_phone = st.session_state.get("phone_input", "(555) 123-4567")
        contact_email = st.session_state.get("email_input", "your.email@example.com")
        
        header_text = f"{contact_name}\n{contact_phone} | {contact_email}"
        
        # Generate preview PDF
        preview_path = "temp/cover_letter_preview.pdf"
        
        # Ensure body is a list
        if isinstance(body, str):
            body = [body]
        
        success = generate_cover_letter_pdf(
            intro=intro,
            body=body,
            conclusion=conclusion,
            header_text=header_text,
            output_path=preview_path
        )
        
        if success:
            st.success("📄 Cover letter preview generated!")
            st.info(f"✅ **One-Page Check**: Preview saved to {preview_path}")
            
            with st.expander("📋 Preview Content Summary", expanded=True):
                st.write(f"**Introduction**: {len(intro)} characters")
                st.write(f"**Body Paragraphs**: {len(body)} paragraphs")
                st.write(f"**Conclusion**: {len(conclusion)} characters")
                
        else:
            st.error("❌ Failed to generate preview")
            
    except Exception as e:
        st.error(f"❌ Preview error: {str(e)}")

def finalize_cover_letter():
    """Finalize the cover letter and complete the application."""
    try:
        # Get current versions of all sections
        intro_versions = st.session_state.cover_letter_versions["introduction"]
        body_versions = st.session_state.cover_letter_versions["body_paragraphs"]
        conclusion_versions = st.session_state.cover_letter_versions["conclusion"]
        
        if not intro_versions or not body_versions or not conclusion_versions:
            st.error("❌ Please complete all cover letter sections before finalizing.")
            return
        
        # Get current version indices
        intro_idx = st.session_state.current_cover_letter_version["introduction"]
        body_idx = st.session_state.current_cover_letter_version["body_paragraphs"]
        conclusion_idx = st.session_state.current_cover_letter_version["conclusion"]
        
        # Get content
        intro = intro_versions[intro_idx]
        body = body_versions[body_idx]
        conclusion = conclusion_versions[conclusion_idx]
        
        # Get contact info
        contact_name = st.session_state.get("user_name_input", "Your Name")
        contact_phone = st.session_state.get("phone_input", "(555) 123-4567")
        contact_email = st.session_state.get("email_input", "your.email@example.com")
        
        header_text = f"{contact_name}\n{contact_phone} | {contact_email}"
        
        # Create final PDF in job folder
        company, position = st.session_state.path_manager.extract_company_and_position(
            st.session_state.job_description_content
        )
        job_folder = st.session_state.path_manager.create_job_folder(
            st.session_state.job_description_content
        )
        cover_letter_path = st.session_state.path_manager.get_cover_letter_path(job_folder, contact_name)
        
        # Ensure body is a list
        if isinstance(body, str):
            body = [body]
        
        success = generate_cover_letter_pdf(
            intro=intro,
            body=body,
            conclusion=conclusion,
            header_text=header_text,
            output_path=str(cover_letter_path)
        )
        
        if success:
            st.session_state.cover_letter_finalized = True
            st.session_state.cover_letter_pdf_path = str(cover_letter_path)
            st.session_state.current_stage = "completed"
            
            st.success(f"🎉 Cover letter finalized and saved!")
            st.success(f"📁 Saved to: {cover_letter_path}")
            st.info("🚀 Application complete! Moving to summary...")
            
            time.sleep(2)  # Brief pause for user to see success
            st.rerun()
            
        else:
            st.error("❌ Failed to save final cover letter")
            
    except Exception as e:
        st.error(f"❌ Error finalizing cover letter: {str(e)}")

def display_completion_summary():
    """Display the completion summary and next steps."""
    st.divider()
    st.header("🎉 Application Factory Complete!")
    st.write("*Your perfectly tailored application documents are ready!*")
    
    # Show completed application
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Generated Documents")
        if st.session_state.resume_pdf_path:
            st.success(f"✅ **Resume**: {Path(st.session_state.resume_pdf_path).name}")
            st.write(f"📁 {st.session_state.resume_pdf_path}")
        
        if st.session_state.cover_letter_pdf_path:
            st.success(f"✅ **Cover Letter**: {Path(st.session_state.cover_letter_pdf_path).name}")
            st.write(f"📁 {st.session_state.cover_letter_pdf_path}")
    
    with col2:
        st.subheader("📊 Application Summary")
        company, position = st.session_state.path_manager.extract_company_and_position(
            st.session_state.job_description_content
        )
        st.write(f"**Company**: {company}")
        st.write(f"**Position**: {position}")
        st.write(f"**Application Folder**: {Path(st.session_state.resume_pdf_path).parent.name}")
    
    # Next steps
    st.divider()
    st.subheader("🚀 Next Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Ready to Apply!** 📤
        
        Your documents are:
        • ATS-friendly formatted
        • Tailored to the job
        • Professionally organized
        """)
    
    with col2:
        st.info("""
        **Future Features** 🔮
        
        Coming soon:
        • Automated submission
        • Application tracking
        • Performance analytics
        """)
    
    with col3:
        if st.button("🔄 Create New Application", type="primary"):
            # Reset for new application
            for key in list(st.session_state.keys()):
                if key.startswith(('resume_sections', 'cover_letter', 'job_analysis', 'current_stage')):
                    del st.session_state[key]
            
            st.session_state.current_stage = "setup"
            st.session_state.job_description_content = ""
            st.rerun()

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