"""
LLM utilities for the Application Factory - Phase 4: LLM Integration & Prompt Engineering.

This module provides intelligent content generation using Google Gemini API,
integrated with RAG for context-aware resume and cover letter creation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import re

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Google AI not available. Install with: pip install google-generativeai")

from config import get_gemini_api_key
from rag_utils import RAGManager

logger = logging.getLogger(__name__)


class PromptTemplates:
    """
    Centralized prompt templates for different LLM tasks.
    """
    
    RESUME_SECTION_PROMPT = """
You are an expert resume writer specializing in ATS-optimized, professional resumes. 

**Task**: Generate a tailored {section_name} section for a resume.

**Context Information**:
- **Job Description**: {job_description}
- **Retrieved Context from Master Resume**: {retrieved_context}
- **Section Type**: {section_name}

**Requirements**:
1. **ATS-Friendly**: Use standard formatting, relevant keywords from job description
2. **Professional Tone**: Clear, concise, action-oriented language
3. **Tailored Content**: Align with job requirements and highlight relevant experience
4. **Quantified Results**: Include metrics and achievements where applicable
5. **Length**: Keep concise but comprehensive

**Format Guidelines**:
- **Summary**: 2-3 sentences highlighting key qualifications
- **Skills**: Categorized list (Programming Languages: X, Y, Z | Tools: A, B, C)
- **Education**: Institution, degree, GPA (if strong), relevant coursework
- **Experience**: Company, role, dates, 3-4 bullet points with achievements
- **Projects**: Project name, 2-3 bullet points with technologies and impact

**Output**: Return only the {section_name} content, ready for PDF generation. No extra commentary.

Generate the {section_name} section:
"""

    COVER_LETTER_PROMPT = """
You are an expert cover letter writer specializing in compelling, personalized cover letters.

**Task**: Generate a tailored cover letter for the specified position.

**Context Information**:
- **Job Description**: {job_description}
- **Company**: {company}
- **Position**: {position}
- **Retrieved Context from Resume**: {retrieved_context}
- **Applicant Name**: {applicant_name}

**Cover Letter Structure**:
1. **Opening**: Express interest in specific position and company
2. **Body Paragraph 1**: Highlight most relevant experience/skills matching job requirements
3. **Body Paragraph 2**: Demonstrate knowledge of company and cultural fit
4. **Body Paragraph 3**: Specific achievements and value proposition
5. **Closing**: Professional closing with call to action

**Requirements**:
- **Personalized**: Reference specific job requirements and company details
- **Professional Tone**: Confident but not arrogant
- **Quantified Achievements**: Include specific metrics and results
- **Company Research**: Show knowledge of company mission/values
- **Length**: 3-4 paragraphs, concise but impactful

**Output Format**:
Return a JSON object with:
{{
    "introduction": "Dear Hiring Manager paragraph...",
    "body_paragraphs": ["paragraph 1", "paragraph 2", "paragraph 3"],
    "conclusion": "Closing paragraph with call to action..."
}}

Generate the cover letter:
"""

    CONTENT_ENHANCEMENT_PROMPT = """
You are a professional content editor specializing in resume and cover letter optimization.

**Task**: Enhance and optimize the provided content for maximum impact.

**Content to Enhance**: {content}
**Content Type**: {content_type}
**Job Context**: {job_description}

**Enhancement Guidelines**:
1. **Clarity**: Improve readability and flow
2. **Impact**: Strengthen action verbs and quantify achievements
3. **Relevance**: Ensure alignment with job requirements
4. **ATS Optimization**: Include relevant keywords naturally
5. **Professional Tone**: Maintain appropriate voice and style

**Focus Areas**:
- Replace weak verbs with strong action verbs
- Add specific metrics and quantified results
- Improve technical terminology alignment
- Enhance readability and conciseness
- Ensure keyword optimization for ATS

**Output**: Return the enhanced content maintaining the original structure and format.

Enhanced content:
"""

    SKILLS_EXTRACTION_PROMPT = """
You are a technical skills analyst specializing in job requirement analysis.

**Task**: Extract and categorize relevant skills from the job description.

**Job Description**: {job_description}

**Categories to Extract**:
1. **Programming Languages**: Specific languages mentioned
2. **Frameworks/Libraries**: Web frameworks, ML libraries, etc.
3. **Tools/Technologies**: Development tools, platforms, databases
4. **Soft Skills**: Communication, leadership, collaboration
5. **Domain Knowledge**: Industry-specific knowledge areas

**Output Format**:
Return a JSON object:
{{
    "programming_languages": ["language1", "language2"],
    "frameworks_libraries": ["framework1", "framework2"],
    "tools_technologies": ["tool1", "tool2"],
    "soft_skills": ["skill1", "skill2"],
    "domain_knowledge": ["domain1", "domain2"],
    "priority_keywords": ["top keywords for ATS"]
}}

Extract skills:
"""


class LLMManager:
    """
    Manager for LLM operations using Google Gemini API.
    Integrates with RAG system for context-aware content generation.
    """
    
    def __init__(self, rag_manager: Optional[RAGManager] = None):
        """
        Initialize LLM Manager.
        
        Args:
            rag_manager: RAG manager for context retrieval
        """
        self.rag_manager = rag_manager
        self.templates = PromptTemplates()
        self.model = None
        self._setup_gemini()
        
    def _setup_gemini(self):
        """Setup Google Gemini API configuration."""
        if not GOOGLE_AI_AVAILABLE:
            logger.error("Google AI SDK not available. Please install with: pip install google-generativeai")
            return
            
        try:
            api_key = get_gemini_api_key()
            if not api_key:
                raise ValueError("Gemini API key not found")
            
            genai.configure(api_key=api_key)
            
            # Configure the model
            self.model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={
                    "temperature": 0.3,  # Lower temperature for more consistent, professional output
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
            
            logger.info("✅ Gemini API configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Gemini API: {e}")
            raise
    
    def _generate_content(self, prompt: str) -> str:
        """
        Generate content using Gemini API.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Generated content as string
        """
        if not self.model:
            raise ValueError("Gemini API not properly configured")
            
        try:
            response = self.model.generate_content(prompt)
            
            if response.text:
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini API")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise
    
    def generate_resume_section(
        self, 
        section_name: str, 
        job_description: str, 
        master_resume_content: str
    ) -> str:
        """
        Generate a tailored resume section using RAG and LLM.
        
        Args:
            section_name: Name of the section (summary, skills, education, etc.)
            job_description: Target job description
            master_resume_content: Full master resume content
            
        Returns:
            Generated section content
        """
        try:
            # Retrieve relevant context using RAG
            retrieved_context = ""
            if self.rag_manager:
                contexts = self.rag_manager.retrieve_for_resume_section(
                    section_name, job_description, master_resume_content
                )
                retrieved_context = "\n".join(contexts) if contexts else "No specific context found"
            
            # Prepare prompt
            prompt = self.templates.RESUME_SECTION_PROMPT.format(
                section_name=section_name,
                job_description=job_description,
                retrieved_context=retrieved_context
            )
            
            # Generate content
            generated_content = self._generate_content(prompt)
            
            logger.info(f"✅ Generated {section_name} section ({len(generated_content)} chars)")
            return generated_content
            
        except Exception as e:
            logger.error(f"Error generating {section_name} section: {e}")
            return f"Error generating {section_name} section. Please try again."
    
    def generate_cover_letter(
        self, 
        job_description: str, 
        company: str, 
        position: str, 
        applicant_name: str,
        master_resume_content: str
    ) -> Dict[str, Any]:
        """
        Generate a tailored cover letter using RAG and LLM.
        
        Args:
            job_description: Target job description
            company: Company name
            position: Position title
            applicant_name: Applicant's name
            master_resume_content: Full master resume content
            
        Returns:
            Dictionary with cover letter components
        """
        try:
            # Retrieve relevant context using RAG
            retrieved_context = ""
            if self.rag_manager:
                contexts = self.rag_manager.retrieve_for_resume_section(
                    "experience", job_description, master_resume_content
                )
                retrieved_context = "\n".join(contexts) if contexts else "No specific context found"
            
            # Prepare prompt
            prompt = self.templates.COVER_LETTER_PROMPT.format(
                job_description=job_description,
                company=company,
                position=position,
                retrieved_context=retrieved_context,
                applicant_name=applicant_name
            )
            
            # Generate content
            generated_content = self._generate_content(prompt)
            
            # Parse JSON response
            try:
                cover_letter_data = json.loads(generated_content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse JSON response, using fallback format")
                cover_letter_data = {
                    "introduction": generated_content[:500],
                    "body_paragraphs": [generated_content[500:1000], generated_content[1000:1500]],
                    "conclusion": generated_content[1500:]
                }
            
            logger.info(f"✅ Generated cover letter for {position} at {company}")
            return cover_letter_data
            
        except Exception as e:
            logger.error(f"Error generating cover letter: {e}")
            return {
                "introduction": f"Dear Hiring Manager,\n\nI am writing to express my interest in the {position} position at {company}.",
                "body_paragraphs": ["Error generating cover letter content. Please try again."],
                "conclusion": f"Thank you for considering my application.\n\nSincerely,\n{applicant_name}"
            }
    
    def enhance_content(self, content: str, content_type: str, job_description: str) -> str:
        """
        Enhance existing content for better impact and ATS optimization.
        
        Args:
            content: Content to enhance
            content_type: Type of content (resume_section, cover_letter, etc.)
            job_description: Job context for enhancement
            
        Returns:
            Enhanced content
        """
        try:
            prompt = self.templates.CONTENT_ENHANCEMENT_PROMPT.format(
                content=content,
                content_type=content_type,
                job_description=job_description
            )
            
            enhanced_content = self._generate_content(prompt)
            
            logger.info(f"✅ Enhanced {content_type} content")
            return enhanced_content
            
        except Exception as e:
            logger.error(f"Error enhancing content: {e}")
            return content  # Return original content on error
    
    def extract_job_skills(self, job_description: str) -> Dict[str, List[str]]:
        """
        Extract and categorize skills from job description.
        
        Args:
            job_description: Job description text
            
        Returns:
            Categorized skills dictionary
        """
        try:
            prompt = self.templates.SKILLS_EXTRACTION_PROMPT.format(
                job_description=job_description
            )
            
            generated_content = self._generate_content(prompt)
            
            # Parse JSON response
            try:
                skills_data = json.loads(generated_content)
            except json.JSONDecodeError:
                logger.warning("Failed to parse skills extraction JSON")
                skills_data = {
                    "programming_languages": [],
                    "frameworks_libraries": [],
                    "tools_technologies": [],
                    "soft_skills": [],
                    "domain_knowledge": [],
                    "priority_keywords": []
                }
            
            logger.info("✅ Extracted job skills and keywords")
            return skills_data
            
        except Exception as e:
            logger.error(f"Error extracting job skills: {e}")
            return {
                "programming_languages": [],
                "frameworks_libraries": [],
                "tools_technologies": [],
                "soft_skills": [],
                "domain_knowledge": [],
                "priority_keywords": []
            }
    
    def generate_complete_resume(
        self, 
        job_description: str, 
        master_resume_content: str,
        sections: List[str] = None
    ) -> Dict[str, str]:
        """
        Generate a complete tailored resume with all sections.
        
        Args:
            job_description: Target job description
            master_resume_content: Full master resume content
            sections: List of sections to generate (default: all)
            
        Returns:
            Dictionary with all generated resume sections
        """
        if sections is None:
            sections = ['summary', 'skills', 'education', 'experience', 'projects']
        
        resume_sections = {}
        
        for section in sections:
            try:
                generated_section = self.generate_resume_section(
                    section, job_description, master_resume_content
                )
                resume_sections[section] = generated_section
            except Exception as e:
                logger.error(f"Failed to generate {section} section: {e}")
                resume_sections[section] = f"Error generating {section} section"
        
        logger.info(f"✅ Generated complete resume with {len(resume_sections)} sections")
        return resume_sections


# Convenience functions for easier access
def create_llm_manager(rag_manager: Optional[RAGManager] = None) -> LLMManager:
    """
    Create and configure an LLM manager.
    
    Args:
        rag_manager: Optional RAG manager for context retrieval
        
    Returns:
        Configured LLM manager
    """
    return LLMManager(rag_manager)


def generate_tailored_resume(
    job_description: str,
    master_resume_content: str,
    rag_manager: Optional[RAGManager] = None
) -> Dict[str, str]:
    """
    Convenience function to generate a complete tailored resume.
    
    Args:
        job_description: Target job description
        master_resume_content: Master resume content
        rag_manager: Optional RAG manager
        
    Returns:
        Dictionary with generated resume sections
    """
    llm_manager = create_llm_manager(rag_manager)
    return llm_manager.generate_complete_resume(job_description, master_resume_content)


def generate_tailored_cover_letter(
    job_description: str,
    company: str,
    position: str,
    applicant_name: str,
    master_resume_content: str,
    rag_manager: Optional[RAGManager] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate a tailored cover letter.
    
    Args:
        job_description: Target job description
        company: Company name
        position: Position title
        applicant_name: Applicant name
        master_resume_content: Master resume content
        rag_manager: Optional RAG manager
        
    Returns:
        Dictionary with cover letter components
    """
    llm_manager = create_llm_manager(rag_manager)
    return llm_manager.generate_cover_letter(
        job_description, company, position, applicant_name, master_resume_content
    ) 