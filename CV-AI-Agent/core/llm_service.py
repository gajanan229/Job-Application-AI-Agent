"""
LLM Service Module for Application Factory

This module provides comprehensive LLM integration using Google's Gemini API
for generating tailored resumes and cover letters based on RAG context.

Classes:
    - LLMService: Main service for content generation
    - PromptManager: Centralized prompt management
    - ContentGenerator: Specialized content generation logic
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from config.settings import config
from config.logging_config import get_logger, timing_decorator
from utils.error_handlers import error_handler, LLMError, ValidationError

logger = get_logger(__name__)


class ContentType(Enum):
    """Enumeration for different content generation types."""
    RESUME = "resume"
    RESUME_SUMMARY = "resume_summary"
    RESUME_PROJECTS = "resume_projects"
    COVER_LETTER = "cover_letter"
    COVER_LETTER_INTRO = "cover_letter_intro"
    COVER_LETTER_BODY = "cover_letter_body"
    COVER_LETTER_CONCLUSION = "cover_letter_conclusion"
    ANALYSIS = "analysis"


@dataclass
class GenerationRequest:
    """Request structure for content generation."""
    content_type: ContentType
    rag_context: List[Dict[str, Any]]
    job_description: str
    master_resume_text: str
    user_preferences: Optional[Dict[str, Any]] = None
    additional_context: Optional[str] = None


@dataclass
class GenerationResponse:
    """Response structure for generated content."""
    content: str
    content_type: ContentType
    metadata: Dict[str, Any]
    generation_time: float
    token_usage: Optional[Dict[str, int]] = None


class PromptManager:
    """Manages all prompts for different content generation tasks."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.PromptManager")
        self._prompts = self._initialize_prompts()
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize all prompt templates."""
        return {
            "resume_system": self._get_resume_system_prompt(),
            "resume_user": self._get_resume_user_prompt(),
            "resume_summary_system": self._get_resume_summary_system_prompt(),
            "resume_summary_user": self._get_resume_summary_user_prompt(),
            "resume_projects_system": self._get_resume_projects_system_prompt(),
            "resume_projects_user": self._get_resume_projects_user_prompt(),
            "cover_letter_system": self._get_cover_letter_system_prompt(),
            "cover_letter_user": self._get_cover_letter_user_prompt(),
            "cover_letter_intro_system": self._get_cover_letter_intro_system_prompt(),
            "cover_letter_intro_user": self._get_cover_letter_intro_user_prompt(),
            "cover_letter_body_system": self._get_cover_letter_body_system_prompt(),
            "cover_letter_body_user": self._get_cover_letter_body_user_prompt(),
            "cover_letter_conclusion_system": self._get_cover_letter_conclusion_system_prompt(),
            "cover_letter_conclusion_user": self._get_cover_letter_conclusion_user_prompt(),
            "analysis_system": self._get_analysis_system_prompt(),
            "analysis_user": self._get_analysis_user_prompt()
        }
    
    def _get_resume_system_prompt(self) -> str:
        """System prompt for resume generation."""
        return """You are an expert resume writer and career counselor with extensive experience in creating compelling, 
ATS-optimized resumes for various industries. Your task is to transform a master resume into a tailored version 
that perfectly matches a specific job description.

KEY RESPONSIBILITIES:
1. Analyze the job description to identify key requirements, skills, and qualifications
2. Extract relevant experience and achievements from the master resume
3. Reframe and optimize content to align with the target role
4. Ensure ATS optimization with appropriate keywords
5. Maintain professional formatting and clear structure

OPTIMIZATION PRINCIPLES:
- Use action verbs and quantifiable achievements
- Incorporate job-specific keywords naturally
- Prioritize most relevant experience
- Maintain truthfulness while optimizing presentation
- Follow standard resume best practices
- Ensure content flows logically and professionally

OUTPUT FORMAT:
Provide a complete, professionally formatted resume in plain text that can be easily transferred to a document template.
Use clear section headers and consistent formatting throughout."""

    def _get_resume_user_prompt(self) -> str:
        """User prompt template for resume generation."""
        return """Please create a tailored resume based on the following information:

**JOB DESCRIPTION:**
{job_description}

**MASTER RESUME CONTENT:**
{master_resume_text}

**RELEVANT CONTEXT FROM RAG ANALYSIS:**
{rag_context}

**ADDITIONAL INSTRUCTIONS:**
{additional_instructions}

Please generate a comprehensive, tailored resume that:
1. Aligns perfectly with the job requirements
2. Highlights the most relevant experiences and skills
3. Uses appropriate keywords from the job description
4. Maintains professional formatting and structure
5. Quantifies achievements where possible
6. Follows ATS best practices

Focus on making the candidate appear as the ideal fit for this specific role while maintaining complete honesty and accuracy."""

    def _get_resume_summary_system_prompt(self) -> str:
        """System prompt for resume summary generation."""
        return """You are an expert resume writer specializing in creating compelling summary sections that capture a candidate's value proposition in 2-3 sentences.

CRITICAL REQUIREMENTS:
- Generate ONLY the summary paragraph text - NO headers, NO contact info, NO other sections
- Output should be 2-3 sentences maximum
- Focus on the candidate's key value proposition for the specific role
- Include relevant keywords from the job description naturally
- Highlight the most impactful qualifications and achievements

WRITING PRINCIPLES:
- Start with the candidate's current status/level (e.g., "Third-year Computer Science student")
- Include the specific role being targeted
- Mention 2-3 most relevant skills/experiences
- Include quantifiable achievements when possible
- End with enthusiasm for the specific opportunity/company

OUTPUT FORMAT:
Return ONLY the plain text summary paragraph - no formatting, no section headers, no extra text."""

    def _get_resume_summary_user_prompt(self) -> str:
        """User prompt template for resume summary generation."""
        return """Generate ONLY a resume summary paragraph (2-3 sentences) based on the following information:

**JOB DESCRIPTION:**
{job_description}

**MASTER RESUME CONTENT:**
{master_resume_text}

**RELEVANT CONTEXT FROM RAG ANALYSIS:**
{rag_context}

**ADDITIONAL INSTRUCTIONS:**
{additional_instructions}

REQUIREMENTS:
- Generate ONLY the summary paragraph text
- 2-3 sentences maximum
- Include the specific role title from the job description
- Highlight the most relevant qualifications from the master resume
- Include relevant keywords naturally
- NO headers, NO contact information, NO other sections

Example format: "Highly motivated [current status] seeking [specific role] position. Possesses [2-3 key qualifications/skills] with experience in [relevant areas]. Eager to contribute to [specific company/role context]."

Generate the summary paragraph now:"""

    def _get_resume_projects_system_prompt(self) -> str:
        """System prompt for resume projects generation."""
        return """You are an expert resume writer specializing in selecting and formatting the most relevant projects from resumes.

CRITICAL REQUIREMENTS:
- Select TOP 3-5 projects that best match the job description AND are technically impressive
- Use EXACT formatting template for each project:
  [PROJECT TITLE]
  ● [PROJECT BULLET POINTS]
- DO NOT fabricate or hallucinate any information
- Extract content exactly as written in master resume
- Prioritize projects by: (1) Job relevance score (2) Technical impressiveness

PROJECT SELECTION CRITERIA:
1. Technical alignment with job requirements (programming languages, frameworks, tools)
2. Technical complexity and impressiveness (architecture, scale, innovation)
3. Quantifiable achievements and impact
4. Relevance to target role responsibilities

FORMATTING REQUIREMENTS:
- Each project must follow this EXACT template:
  [PROJECT TITLE]
  ● [BULLET POINT 1]
  ● [BULLET POINT 2]
  ● [BULLET POINT 3]
  (continue with all original bullet points)

- Preserve original bullet point content exactly
- Maintain original technology lists and metrics
- Keep projects concise to fit 1-page resume limit

OUTPUT FORMAT:
Return ONLY the formatted projects using the exact template above, limited to 3-5 best projects."""

    def _get_resume_projects_user_prompt(self) -> str:
        """User prompt template for resume projects generation."""
        return """Extract and present the projects from the master resume, ordered by relevance to the job description.

**JOB DESCRIPTION:**
{job_description}

**MASTER RESUME CONTENT:**
{master_resume_text}

**RELEVANT CONTEXT FROM RAG ANALYSIS:**
{rag_context}

**ADDITIONAL INSTRUCTIONS:**
{additional_instructions}

REQUIREMENTS:
- Select TOP 3-5 projects that best match job description AND are technically impressive
- Use EXACT formatting template for each selected project:
  [PROJECT TITLE]
  ● [BULLET POINT 1]
  ● [BULLET POINT 2]
  ● [etc...]
- DO NOT modify, add, or fabricate any project content
- Preserve original bullet point text exactly as written
- Prioritize by: (1) Job relevance (2) Technical impressiveness
- Ensure final output fits within 1-page resume limit

SCORING CRITERIA:
1. Technical skills match (languages, frameworks, tools mentioned in job)
2. Project complexity and scale (architecture, user base, performance)  
3. Quantifiable achievements (metrics, improvements, impact)
4. Relevance to target role responsibilities

Return ONLY the formatted top 3-5 projects using the exact template format:"""

    def _get_cover_letter_system_prompt(self) -> str:
        """System prompt for cover letter generation."""
        return """You are an expert cover letter writer with a proven track record of helping candidates secure interviews 
at top companies. Your task is to create compelling, personalized cover letters that effectively communicate 
a candidate's value proposition for specific roles.

KEY RESPONSIBILITIES:
1. Craft engaging opening that captures attention
2. Demonstrate clear understanding of the role and company
3. Highlight most relevant qualifications and achievements
4. Show genuine enthusiasm and cultural fit
5. Create strong closing with clear next steps
6. Maintain professional yet personable tone

WRITING PRINCIPLES:
- Open with impact, not generic statements
- Use specific examples and quantified achievements
- Show research and genuine interest in the company
- Address potential concerns proactively
- Create emotional connection while staying professional
- End with confidence and clear call to action

STRUCTURE GUIDELINES:
- Professional header with contact information
- Personalized salutation when possible
- Compelling opening paragraph
- 2-3 substantive body paragraphs
- Strong closing paragraph
- Professional sign-off

OUTPUT FORMAT:
Provide a complete, professionally formatted cover letter in plain text that demonstrates clear value alignment 
between the candidate and the opportunity."""

    def _get_cover_letter_user_prompt(self) -> str:
        """User prompt template for cover letter generation."""
        return """Please create a compelling cover letter based on the following information:

**JOB DESCRIPTION:**
{job_description}

**MASTER RESUME CONTENT:**
{master_resume_text}

**RELEVANT CONTEXT FROM RAG ANALYSIS:**
{rag_context}

**ADDITIONAL INSTRUCTIONS:**
{additional_instructions}

Please generate a professional cover letter that:
1. Opens with a strong, engaging hook
2. Demonstrates clear understanding of the role and company needs
3. Highlights 2-3 most relevant achievements with specific examples
4. Shows genuine enthusiasm for the opportunity
5. Addresses any potential gaps or concerns
6. Closes with confidence and clear next steps
7. Maintains professional yet personable tone throughout

The cover letter should feel authentic and personalized, not generic. Focus on creating a compelling narrative 
that positions the candidate as the ideal solution to the company's needs."""

    def _get_cover_letter_intro_system_prompt(self) -> str:
        """System prompt for cover letter introduction generation."""
        return """You are an expert cover letter writer specializing in creating compelling opening paragraphs that immediately capture the reader's attention.

CRITICAL REQUIREMENTS:
- Generate ONLY the introduction paragraph - NO headers, NO addresses, NO "Dear [Name]"
- Create a unique, engaging hook that stands out from generic openings
- Length: 3-4 sentences maximum
- Show genuine company/role research and enthusiasm
- Establish immediate credibility and relevance

WRITING PRINCIPLES:
- Start with creative hook (analogy, insight, specific company detail)
- Connect personal passion/experience to company mission
- Mention specific role title and company name
- Highlight 1-2 key qualifications briefly
- End with forward-looking enthusiasm

AVOID:
- Generic openings like "I am writing to apply..."
- Listing qualifications without context
- Overly formal or stuffy language
- Clichés or overused phrases

OUTPUT FORMAT:
Return ONLY the introduction paragraph in plain text - no formatting, no additional text."""

    def _get_cover_letter_intro_user_prompt(self) -> str:
        """User prompt template for cover letter introduction generation."""
        return """Generate ONLY a compelling cover letter introduction paragraph based on the following information:

**JOB DESCRIPTION:**
{job_description}

**MASTER RESUME CONTENT:**
{master_resume_text}

**RELEVANT CONTEXT FROM RAG ANALYSIS:**
{rag_context}

**ADDITIONAL INSTRUCTIONS:**
{additional_instructions}

REQUIREMENTS:
- Generate ONLY the introduction paragraph (3-4 sentences)
- NO headers, NO addresses, NO "Dear [Name]", NO extra text
- Start with a unique, creative hook that relates to the company/role
- Mention the specific company name and role title
- Briefly highlight 1-2 key qualifications
- Show genuine enthusiasm and research about the company
- End with forward-looking statement about contributing

Example style: "In the vast ocean of [industry context], [creative analogy/insight]. As a [candidate status] who's [relevant experience], I was thrilled to discover [Company]'s [specific role] position. Like [company reference/analogy], I've [relevant qualification] while [another strength]."

Generate the introduction paragraph now:"""

    def _get_cover_letter_body_system_prompt(self) -> str:
        """System prompt for cover letter body generation."""
        return """You are an expert cover letter writer specializing in creating compelling body paragraphs that showcase candidate qualifications through specific examples.

CRITICAL REQUIREMENTS:
- Generate EXACTLY 2 concise body paragraphs
- Each paragraph must be 3-4 sentences maximum (60-80 words each)
- Focus on different aspects of qualifications
- Use specific examples with quantified achievements
- Connect candidate experience directly to job requirements
- Keep content under 160 words total for both paragraphs

PARAGRAPH STRUCTURE:
Paragraph 1 (3-4 sentences): Technical skills and most relevant project experience
Paragraph 2 (3-4 sentences): Achievements, problem-solving abilities, and cultural fit

WRITING PRINCIPLES:
- Start each paragraph with strong, concise topic sentence
- Include ONE specific example per paragraph with metrics
- Show direct relevance to job requirements
- Be concise but impactful - every word counts
- Focus on the most impressive and relevant qualifications only

LENGTH CONSTRAINTS:
- Paragraph 1: 60-80 words maximum
- Paragraph 2: 60-80 words maximum  
- Total: Under 160 words for both paragraphs combined

OUTPUT FORMAT:
Return ONLY the 2 body paragraphs in plain text - no introduction, no conclusion, no extra formatting."""

    def _get_cover_letter_body_user_prompt(self) -> str:
        """User prompt template for cover letter body generation."""
        return """Generate 2-3 compelling cover letter body paragraphs based on the following information:

**JOB DESCRIPTION:**
{job_description}

**MASTER RESUME CONTENT:**
{master_resume_text}

**RELEVANT CONTEXT FROM RAG ANALYSIS:**
{rag_context}

**INTRODUCTION CONTEXT:**
{additional_context}

**ADDITIONAL INSTRUCTIONS:**
{additional_instructions}

REQUIREMENTS:
- Generate EXACTLY 2 concise body paragraphs
- Each paragraph must be 3-4 sentences maximum (60-80 words each)
- Total word count under 160 words for both paragraphs combined
- Use specific examples from the candidate's experience with quantified results
- Connect each example directly to specific job requirements
- Flow naturally from the introduction context provided

PARAGRAPH FOCUS:
1. Technical skills and most relevant project experience (60-80 words)
2. Achievements, problem-solving abilities, and cultural fit (60-80 words)

LENGTH VALIDATION:
- Count words carefully to stay under limits
- Prioritize impact over length
- Use concise, powerful language

Generate the body paragraphs now:"""

    def _get_cover_letter_conclusion_system_prompt(self) -> str:
        """System prompt for cover letter conclusion generation."""
        return """You are an expert cover letter writer specializing in creating powerful closing paragraphs that leave a lasting impression.

CRITICAL REQUIREMENTS:
- Generate ONLY the conclusion paragraph - NO signatures, NO extra text
- Length: 2-3 sentences maximum
- Express confidence in ability to contribute
- Include clear call to action for next steps
- Maintain enthusiasm while being professional

WRITING PRINCIPLES:
- Start with forward-looking confidence statement
- Reference specific value you'll bring to the team/company
- End with professional call to action
- Express gratitude professionally
- Avoid generic phrases like "I look forward to hearing from you"

CLOSING STRUCTURE:
1. Confident statement about readiness to contribute
2. Specific mention of team/company mission
3. Professional call to action and gratitude

OUTPUT FORMAT:
Return ONLY the conclusion paragraph in plain text - no signature lines, no additional formatting."""

    def _get_cover_letter_conclusion_user_prompt(self) -> str:
        """User prompt template for cover letter conclusion generation."""
        return """Generate ONLY a strong cover letter conclusion paragraph based on the following information:

**JOB DESCRIPTION:**
{job_description}

**MASTER RESUME CONTENT:**
{master_resume_text}

**RELEVANT CONTEXT FROM RAG ANALYSIS:**
{rag_context}

**ADDITIONAL INSTRUCTIONS:**
{additional_instructions}

REQUIREMENTS:
- Generate ONLY the conclusion paragraph (2-3 sentences)
- NO signature lines, NO "Sincerely", NO extra text
- Express confidence about contributing to the specific team/company
- Include clear call to action for next steps
- Reference the company's mission or specific team mentioned in job description
- Professional gratitude without being overly formal

Example style: "I'm ready to [specific contribution] and contribute meaningfully to [specific team/mission]. I would welcome the opportunity to discuss how my [key strength] could benefit [company goal]. Thank you for considering my application."

Generate the conclusion paragraph now:"""

    def _get_analysis_system_prompt(self) -> str:
        """System prompt for document analysis."""
        return """You are an expert document analyzer specializing in resume and job description analysis. 
Your task is to provide comprehensive insights about document alignment, keyword optimization, 
and improvement recommendations.

ANALYSIS AREAS:
1. Content alignment between resume and job description
2. Keyword presence and optimization opportunities
3. Skills gap identification
4. Experience relevance assessment
5. ATS optimization potential
6. Formatting and structure evaluation

OUTPUT REQUIREMENTS:
- Provide specific, actionable insights
- Include numerical scores where applicable
- Highlight strengths and improvement areas
- Suggest concrete optimization strategies
- Maintain objective, professional analysis"""

    def _get_analysis_user_prompt(self) -> str:
        """User prompt template for document analysis."""
        return """Please analyze the following documents and provide comprehensive insights:

**JOB DESCRIPTION:**
{job_description}

**RESUME CONTENT:**
{master_resume_text}

**RAG ANALYSIS CONTEXT:**
{rag_context}

Please provide:
1. Overall alignment score (1-10)
2. Key strengths in the current resume
3. Identified gaps or missing elements
4. Keyword optimization opportunities
5. Specific recommendations for improvement
6. ATS optimization suggestions

Focus on providing actionable insights that would help improve the candidate's competitiveness for this specific role."""

    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """Get formatted prompt with provided parameters."""
        try:
            if prompt_type not in self._prompts:
                raise ValidationError(f"Unknown prompt type: {prompt_type}")
            
            prompt_template = self._prompts[prompt_type]
            
            # Provide default values for optional parameters
            kwargs.setdefault('additional_context', '')
            
            # Format the prompt with provided kwargs
            if kwargs:
                return prompt_template.format(**kwargs)
            return prompt_template
            
        except KeyError as e:
            raise ValidationError(f"Missing required parameter for prompt: {e}")
        except Exception as e:
            raise LLMError(f"Error formatting prompt: {str(e)}")


class LLMService:
    """Main service for LLM operations using Google Gemini."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM service with Google Gemini."""
        self.logger = get_logger(f"{__name__}.LLMService")
        self.api_key = api_key or config.google_api_key
        self.prompt_manager = PromptManager()
        self.model = None
        self._initialize_client()
    
    @error_handler(reraise=True)
    def _initialize_client(self):
        """Initialize Google Gemini client."""
        if not self.api_key:
            raise LLMError("Google API key is required for LLM service")
        
        try:
            # Configure the API key
            genai.configure(api_key=self.api_key)
            
            # Initialize the model with safety settings
            generation_config = genai.types.GenerationConfig(
                temperature=config.llm_temperature,
                max_output_tokens=config.llm_max_tokens,
                top_p=0.95,
                top_k=40
            )
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            self.model = genai.GenerativeModel(
                model_name=config.llm_model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self.logger.info(f"LLM service initialized successfully with model: {config.llm_model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise LLMError(f"Failed to initialize LLM service: {str(e)}")
    
    @timing_decorator
    @error_handler(reraise=True)
    def generate_content(self, request: GenerationRequest) -> GenerationResponse:
        """Generate content based on the request."""
        start_time = time.time()
        
        try:
            # Prepare prompts based on content type
            system_prompt, user_prompt = self._prepare_prompts(request)
            
            # Generate content using Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            self.logger.info(f"Generating {request.content_type.value} content...")
            response = self.model.generate_content(full_prompt)
            
            if not response.text:
                raise LLMError("Empty response from LLM")
            
            generation_time = time.time() - start_time
            
            # Create response object
            result = GenerationResponse(
                content=response.text.strip(),
                content_type=request.content_type,
                metadata={
                    "model": config.llm_model_name,
                    "temperature": config.llm_temperature,
                    "max_tokens": config.llm_max_tokens,
                    "rag_chunks_used": len(request.rag_context),
                    "prompt_length": len(full_prompt)
                },
                generation_time=generation_time,
                token_usage=self._extract_token_usage(response)
            )
            
            self.logger.info(f"Successfully generated {request.content_type.value} content in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Content generation failed: {str(e)}")
            raise LLMError(f"Content generation failed: {str(e)}")
    
    def _prepare_prompts(self, request: GenerationRequest) -> Tuple[str, str]:
        """Prepare system and user prompts for the request."""
        content_type = request.content_type.value
        
        # Format RAG context for inclusion in prompts
        rag_context_text = self._format_rag_context(request.rag_context)
        
        # Prepare additional instructions
        additional_instructions = ""
        if request.user_preferences:
            additional_instructions = "\n".join([
                f"- {key}: {value}" for key, value in request.user_preferences.items()
            ])
        if request.additional_context:
            additional_instructions += f"\n- Additional Context: {request.additional_context}"
        
        # Always provide additional_context for prompts that need it
        additional_context = request.additional_context or ""
        
        # Get prompts
        system_prompt = self.prompt_manager.get_prompt(f"{content_type}_system")
        user_prompt = self.prompt_manager.get_prompt(
            f"{content_type}_user",
            job_description=request.job_description,
            master_resume_text=request.master_resume_text,
            rag_context=rag_context_text,
            additional_instructions=additional_instructions or "None",
            additional_context=additional_context
        )
        
        return system_prompt, user_prompt
    
    def _format_rag_context(self, rag_context: List[Dict[str, Any]]) -> str:
        """Format RAG context for inclusion in prompts."""
        if not rag_context:
            return "No additional context available."
        
        formatted_chunks = []
        for i, chunk in enumerate(rag_context[:5], 1):  # Limit to top 5 chunks
            content = chunk.get('content', '')
            score = chunk.get('similarity_score', 0)
            source = chunk.get('source', 'Unknown')
            
            formatted_chunks.append(f"""
Relevant Content #{i} (Similarity: {score:.3f}, Source: {source}):
{content[:500]}{'...' if len(content) > 500 else ''}
""")
        
        return "\n".join(formatted_chunks)
    
    def _extract_token_usage(self, response) -> Optional[Dict[str, int]]:
        """Extract token usage information from response."""
        try:
            if hasattr(response, 'usage_metadata'):
                return {
                    'prompt_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                    'completion_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                    'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0)
                }
        except Exception:
            pass
        return None


class ContentGenerator:
    """High-level content generation orchestrator."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize content generator with LLM service."""
        self.logger = get_logger(f"{__name__}.ContentGenerator")
        self.llm_service = LLMService(api_key)
    
    @timing_decorator
    @error_handler(reraise=True)
    def generate_resume(
        self, 
        job_description: str, 
        master_resume_text: str, 
        rag_context: List[Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> GenerationResponse:
        """Generate a tailored resume."""
        request = GenerationRequest(
            content_type=ContentType.RESUME,
            rag_context=rag_context,
            job_description=job_description,
            master_resume_text=master_resume_text,
            user_preferences=user_preferences
        )
        
        return self.llm_service.generate_content(request)
    
    @timing_decorator
    @error_handler(reraise=True)
    def generate_cover_letter(
        self, 
        job_description: str, 
        master_resume_text: str, 
        rag_context: List[Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> GenerationResponse:
        """Generate a tailored cover letter."""
        request = GenerationRequest(
            content_type=ContentType.COVER_LETTER,
            rag_context=rag_context,
            job_description=job_description,
            master_resume_text=master_resume_text,
            user_preferences=user_preferences
        )
        
        return self.llm_service.generate_content(request)
    
    @timing_decorator
    @error_handler(reraise=True)
    def analyze_documents(
        self, 
        job_description: str, 
        master_resume_text: str, 
        rag_context: List[Dict[str, Any]]
    ) -> GenerationResponse:
        """Analyze document alignment and provide recommendations."""
        request = GenerationRequest(
            content_type=ContentType.ANALYSIS,
            rag_context=rag_context,
            job_description=job_description,
            master_resume_text=master_resume_text
        )
        
        return self.llm_service.generate_content(request)
    
    @error_handler(reraise=True)
    def batch_generate(
        self,
        job_description: str,
        master_resume_text: str,
        rag_context: List[Dict[str, Any]],
        content_types: List[ContentType],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[ContentType, GenerationResponse]:
        """Generate multiple content types in batch."""
        results = {}
        
        for content_type in content_types:
            self.logger.info(f"Generating {content_type.value}...")
            
            if content_type == ContentType.RESUME:
                results[content_type] = self.generate_resume(
                    job_description, master_resume_text, rag_context, user_preferences
                )
            elif content_type == ContentType.COVER_LETTER:
                results[content_type] = self.generate_cover_letter(
                    job_description, master_resume_text, rag_context, user_preferences
                )
            elif content_type == ContentType.ANALYSIS:
                results[content_type] = self.analyze_documents(
                    job_description, master_resume_text, rag_context
                )
        
        self.logger.info(f"Batch generation completed for {len(results)} content types")
        return results
    
    @timing_decorator
    @error_handler(reraise=True)
    def generate_resume_summary(
        self, 
        job_description: str, 
        master_resume_text: str, 
        rag_context: List[Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> GenerationResponse:
        """Generate only the resume summary section."""
        request = GenerationRequest(
            content_type=ContentType.RESUME_SUMMARY,
            rag_context=rag_context,
            job_description=job_description,
            master_resume_text=master_resume_text,
            user_preferences=user_preferences
        )
        
        return self.llm_service.generate_content(request)
    
    @timing_decorator
    @error_handler(reraise=True)
    def extract_resume_projects(
        self, 
        job_description: str, 
        master_resume_text: str, 
        rag_context: List[Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> GenerationResponse:
        """Extract and present projects from master resume."""
        request = GenerationRequest(
            content_type=ContentType.RESUME_PROJECTS,
            rag_context=rag_context,
            job_description=job_description,
            master_resume_text=master_resume_text,
            user_preferences=user_preferences
        )
        
        return self.llm_service.generate_content(request)
    
    @timing_decorator
    @error_handler(reraise=True)
    def generate_cover_letter_intro(
        self, 
        job_description: str, 
        master_resume_text: str, 
        rag_context: List[Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> GenerationResponse:
        """Generate only the cover letter introduction."""
        request = GenerationRequest(
            content_type=ContentType.COVER_LETTER_INTRO,
            rag_context=rag_context,
            job_description=job_description,
            master_resume_text=master_resume_text,
            user_preferences=user_preferences
        )
        
        return self.llm_service.generate_content(request)
    
    @timing_decorator
    @error_handler(reraise=True)
    def generate_cover_letter_body(
        self, 
        job_description: str, 
        master_resume_text: str, 
        rag_context: List[Dict[str, Any]],
        intro_text: str = "",
        conclusion_text: str = "",
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> GenerationResponse:
        """Generate the cover letter body with intro/conclusion context."""
        additional_context = f"Introduction context: {intro_text}\nConclusion context: {conclusion_text}"
        
        request = GenerationRequest(
            content_type=ContentType.COVER_LETTER_BODY,
            rag_context=rag_context,
            job_description=job_description,
            master_resume_text=master_resume_text,
            user_preferences=user_preferences,
            additional_context=additional_context
        )
        
        return self.llm_service.generate_content(request)
    
    @timing_decorator
    @error_handler(reraise=True)
    def generate_cover_letter_conclusion(
        self, 
        job_description: str, 
        master_resume_text: str, 
        rag_context: List[Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> GenerationResponse:
        """Generate only the cover letter conclusion."""
        request = GenerationRequest(
            content_type=ContentType.COVER_LETTER_CONCLUSION,
            rag_context=rag_context,
            job_description=job_description,
            master_resume_text=master_resume_text,
            user_preferences=user_preferences
        )
        
        return self.llm_service.generate_content(request)