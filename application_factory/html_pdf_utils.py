"""
HTML-to-PDF generation utilities for the Application Factory.

Provides professional PDF generation for resumes and cover letters using
HTML templates and WeasyPrint for conversion, maintaining exact formatting
from the provided example HTML files.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re

try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

logger = logging.getLogger(__name__)


class HTMLtoPDFGenerator:
    """
    Professional PDF generator using HTML templates and WeasyPrint.
    Maintains exact formatting from the original example HTML files.
    """
    
    def __init__(self, templates_dir: str = None):
        """
        Initialize the HTML-to-PDF generator.
        
        Args:
            templates_dir: Directory containing HTML templates
        """
        if not WEASYPRINT_AVAILABLE:
            raise ImportError("WeasyPrint not available. Install with: pip install weasyprint")
        
        if not JINJA2_AVAILABLE:
            raise ImportError("Jinja2 not available. Install with: pip install jinja2")
        
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True
        )
        
        # Font configuration for WeasyPrint
        self.font_config = FontConfiguration()
        
        logger.info(f"✅ HTML-to-PDF generator initialized with templates from {self.templates_dir}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and prepare text for HTML rendering.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text safe for HTML
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Convert line breaks to HTML breaks
        text = text.replace('\n', '<br>')
        
        return text.strip()
    
    def _format_phone(self, phone: str) -> str:
        """Format phone number consistently."""
        if not phone:
            return ""
        
        # Remove all non-digits
        digits = re.sub(r'\D', '', phone)
        
        # Format as (XXX) XXX-XXXX if we have 10 digits
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        
        return phone  # Return as-is if not 10 digits
    
    def _prepare_resume_context(
        self, 
        sections: Dict[str, str], 
        name: str = "", 
        phone: str = "", 
        email: str = "", 
        linkedin: str = "", 
        website: str = "", 
        location: str = "Toronto, ON"
    ) -> Dict[str, Any]:
        """
        Prepare context variables for resume template.
        
        Args:
            sections: Resume sections content
            name: Full name
            phone: Phone number
            email: Email address
            linkedin: LinkedIn URL
            website: Website URL
            location: Location
            
        Returns:
            Dictionary with template context
        """
        context = {
            'name': name or "Your Name",
            'phone': self._format_phone(phone),
            'email': email or "your.email@example.com",
            'linkedin': linkedin,
            'website': website,
            'website_display': website.replace('https://', '').replace('http://', '') if website else "",
            'location': location,
        }
        
        # Add sections
        for section_name, content in sections.items():
            if content and content.strip():
                if section_name == 'skills':
                    # Process skills section to maintain bullet formatting
                    context[section_name] = self._format_skills_for_html(content)
                elif section_name in ['education', 'experience', 'projects']:
                    # Process structured sections
                    context[section_name] = self._format_section_for_html(content, section_name)
                else:
                    # Summary and other sections
                    context[section_name] = self._clean_text(content)
        
        return context
    
    def _format_skills_for_html(self, skills_content: str) -> str:
        """
        Format skills section for HTML template.
        
        Args:
            skills_content: Raw skills content
            
        Returns:
            Formatted HTML string
        """
        lines = [line.strip() for line in skills_content.split('\n') if line.strip()]
        formatted_lines = []
        
        for line in lines:
            if ':' in line:
                # Category line like "Programming Languages: Python, Java..."
                parts = line.split(':', 1)
                category = parts[0].strip()
                skills = parts[1].strip()
                formatted_line = f"<span class=\"c3\">{category}:</span><span class=\"c2\"> </span><span class=\"c3\">Python</span><span class=\"c0\">{skills}</span>"
                formatted_lines.append(formatted_line)
            else:
                # Regular skill line
                clean_line = line.strip()
                if clean_line.startswith(('*', '•', '-')):
                    clean_line = clean_line[1:].strip()
                formatted_lines.append(f"<span class=\"c0\">{clean_line}</span>")
        
        return '\n'.join(formatted_lines)
    
    def _format_section_for_html(self, content: str, section_type: str) -> str:
        """
        Format education, experience, or projects section for HTML.
        
        Args:
            content: Raw section content
            section_type: Type of section (education, experience, projects)
            
        Returns:
            Formatted HTML string
        """
        if section_type == 'education':
            return self._format_education_html(content)
        elif section_type == 'experience':
            return self._format_experience_html(content)
        elif section_type == 'projects':
            return self._format_projects_html(content)
        else:
            return self._clean_text(content)
    
    def _format_education_html(self, content: str) -> str:
        """Format education section for HTML."""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        html_parts = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['university', 'college', 'institute']):
                # University name
                html_parts.append(f'<p class="c11"><span class="c3">{line}</span><span class="c0"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Toronto, ON</span></p>')
            elif 'bachelor' in line.lower() or 'degree' in line.lower():
                # Degree line
                if 'gpa' in line.lower():
                    parts = line.split('GPA')
                    degree_part = parts[0].strip()
                    gpa_part = 'GPA' + parts[1].strip() if len(parts) > 1 else ''
                    html_parts.append(f'<p class="c19"><span class="c2">{degree_part} </span><span class="c3">{gpa_part}</span><span class="c0"> &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Sep 2023 - Apr 2027</span></p>')
                else:
                    html_parts.append(f'<p class="c19"><span class="c0">{line}</span></p>')
            elif 'relevant coursework' in line.lower():
                html_parts.append(f'<p class="c19"><span class="c3">{line}</span></p>')
            elif line.startswith(('*', '•', '-')):
                # Coursework bullet points
                clean_line = line[1:].strip()
                html_parts.append(f'<li class="c11 c18 li-bullet-0"><span class="c0">{clean_line}</span></li>')
            else:
                html_parts.append(f'<p class="c11"><span class="c0">{line}</span></p>')
        
        return '\n'.join(html_parts)
    
    def _format_experience_html(self, content: str) -> str:
        """Format work experience section for HTML."""
        entries = content.split('\n\n')
        html_parts = []
        
        for entry in entries:
            if not entry.strip():
                continue
            
            lines = [line.strip() for line in entry.split('\n') if line.strip()]
            if not lines:
                continue
            
            # Company name
            if lines:
                html_parts.append(f'<p class="c4"><span class="c3">{lines[0]}</span></p>')
            
            # Job title and dates
            if len(lines) > 1:
                html_parts.append(f'<p class="c7"><span class="c0">{lines[1]}</span></p>')
            
            # Bullet points
            html_parts.append('<ul class="c6 lst-kix_list_1-0 start">')
            for line in lines[2:]:
                if line.strip():
                    clean_line = line.strip()
                    if clean_line.startswith(('*', '•', '-')):
                        clean_line = clean_line[1:].strip()
                    html_parts.append(f'<li class="c16 li-bullet-0"><span class="c0">{clean_line}</span></li>')
            html_parts.append('</ul>')
        
        return '\n'.join(html_parts)
    
    def _format_projects_html(self, content: str) -> str:
        """Format projects section for HTML."""
        entries = content.split('\n\n')
        html_parts = []
        
        for entry in entries:
            if not entry.strip():
                continue
            
            lines = [line.strip() for line in entry.split('\n') if line.strip()]
            if not lines:
                continue
            
            # Project title
            if lines:
                html_parts.append(f'<p class="c26"><span class="c3">{lines[0]}</span></p>')
            
            # Project bullet points
            html_parts.append('<ul class="c6 lst-kix_list_1-0 start">')
            for line in lines[1:]:
                if line.strip():
                    clean_line = line.strip()
                    if clean_line.startswith(('*', '•', '-')):
                        clean_line = clean_line[1:].strip()
                    html_parts.append(f'<li class="c9 li-bullet-0"><span class="c0">{clean_line}</span></li>')
            html_parts.append('</ul>')
        
        return '\n'.join(html_parts)
    
    def _prepare_cover_letter_context(
        self,
        introduction: str,
        body_paragraphs: List[str],
        conclusion: str,
        name: str = "",
        phone: str = "",
        email: str = "",
        linkedin: str = "",
        github: str = "",
        website: str = "",
        location: str = "Toronto, ON"
    ) -> Dict[str, Any]:
        """
        Prepare context variables for cover letter template.
        
        Args:
            introduction: Introduction paragraph
            body_paragraphs: List of body paragraphs
            conclusion: Conclusion paragraph
            name: Full name
            phone: Phone number
            email: Email address
            linkedin: LinkedIn URL
            github: GitHub URL
            website: Website URL
            location: Location
            
        Returns:
            Dictionary with template context
        """
        return {
            'name': name or "Your Name",
            'phone': self._format_phone(phone),
            'email': email or "your.email@example.com",
            'linkedin': linkedin,
            'github': github,
            'website': website,
            'website_display': website.replace('https://', '').replace('http://', '') if website else "",
            'location': location,
            'date': datetime.now().strftime("%B %d, %Y"),
            'introduction': self._clean_text(introduction),
            'body_paragraphs': [self._clean_text(para) for para in body_paragraphs if para.strip()],
            'conclusion': self._clean_text(conclusion)
        }
    
    def generate_resume_pdf(
        self, 
        sections: Dict[str, str], 
        name: str = "",
        phone: str = "",
        email: str = "",
        linkedin: str = "",
        website: str = "",
        location: str = "Toronto, ON",
        output_path: str = "resume.pdf"
    ) -> bool:
        """
        Generate a professional resume PDF using HTML template.
        
        Args:
            sections: Resume sections content
            name: Full name
            phone: Phone number
            email: Email address
            linkedin: LinkedIn URL
            website: Website URL
            location: Location
            output_path: Path for the output PDF
            
        Returns:
            bool: True if successful
        """
        try:
            # Load template
            template = self.jinja_env.get_template('resume_template.html')
            
            # Prepare context
            context = self._prepare_resume_context(
                sections, name, phone, email, linkedin, website, location
            )
            
            # Render HTML
            html_content = template.render(**context)
            
            # Generate PDF
            html_doc = HTML(string=html_content)
            html_doc.write_pdf(output_path, font_config=self.font_config)
            
            # Validate output
            if self._validate_pdf_output(output_path):
                logger.info(f"✅ Resume PDF generated successfully: {output_path}")
                return True
            else:
                logger.warning(f"⚠️ Resume PDF generated but validation failed: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error generating resume PDF: {e}")
            return False
    
    def generate_cover_letter_pdf(
        self,
        introduction: str,
        body_paragraphs: List[str],
        conclusion: str,
        name: str = "",
        phone: str = "",
        email: str = "",
        linkedin: str = "",
        github: str = "",
        website: str = "",
        location: str = "Toronto, ON",
        output_path: str = "cover_letter.pdf"
    ) -> bool:
        """
        Generate a professional cover letter PDF using HTML template.
        
        Args:
            introduction: Introduction paragraph
            body_paragraphs: List of body paragraphs
            conclusion: Conclusion paragraph
            name: Full name
            phone: Phone number
            email: Email address
            linkedin: LinkedIn URL
            github: GitHub URL
            website: Website URL
            location: Location
            output_path: Path for the output PDF
            
        Returns:
            bool: True if successful
        """
        try:
            # Load template
            template = self.jinja_env.get_template('cover_letter_template.html')
            
            # Prepare context
            context = self._prepare_cover_letter_context(
                introduction, body_paragraphs, conclusion,
                name, phone, email, linkedin, github, website, location
            )
            
            # Render HTML
            html_content = template.render(**context)
            
            # Generate PDF
            html_doc = HTML(string=html_content)
            html_doc.write_pdf(output_path, font_config=self.font_config)
            
            # Validate output
            if self._validate_pdf_output(output_path):
                logger.info(f"✅ Cover letter PDF generated successfully: {output_path}")
                return True
            else:
                logger.warning(f"⚠️ Cover letter PDF generated but validation failed: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error generating cover letter PDF: {e}")
            return False
    
    def _validate_pdf_output(self, pdf_path: str) -> bool:
        """
        Validate that the PDF was generated successfully.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            bool: True if valid
        """
        try:
            if not os.path.exists(pdf_path):
                return False
            
            file_size = os.path.getsize(pdf_path)
            
            # Check for reasonable file size (>1KB and <10MB)
            if 1000 < file_size < 10 * 1024 * 1024:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating PDF: {e}")
            return False


# Convenience functions for backward compatibility
def generate_resume_pdf(
    sections: Dict[str, str],
    name: str = "",
    phone: str = "",
    email: str = "",
    linkedin: str = "",
    website: str = "",
    location: str = "Toronto, ON",
    output_path: str = "resume.pdf"
) -> bool:
    """
    Convenience function to generate a resume PDF using HTML templates.
    
    Args:
        sections: Resume sections content
        name: Full name
        phone: Phone number
        email: Email address
        linkedin: LinkedIn URL
        website: Website URL
        location: Location
        output_path: Output file path
        
    Returns:
        bool: True if successful
    """
    generator = HTMLtoPDFGenerator()
    return generator.generate_resume_pdf(
        sections, name, phone, email, linkedin, website, location, output_path
    )


def generate_cover_letter_pdf(
    intro: str,
    body: List[str],
    conclusion: str,
    header_text: Optional[str] = None,
    output_path: str = "cover_letter.pdf"
) -> bool:
    """
    Convenience function to generate a cover letter PDF using HTML templates.
    
    Args:
        intro: Introduction paragraph
        body: Body paragraphs
        conclusion: Conclusion paragraph
        header_text: Header text (parsed for contact info)
        output_path: Output file path
        
    Returns:
        bool: True if successful
    """
    # Parse header_text if provided
    name = "Your Name"
    phone = "(555) 123-4567"
    email = "your.email@example.com"
    
    if header_text:
        lines = header_text.split('\n')
        if lines:
            name = lines[0].strip()
        if len(lines) > 1:
            contact_line = lines[1].strip()
            # Simple parsing - in practice you might want more robust parsing
            parts = contact_line.split('|')
            if len(parts) >= 2:
                phone = parts[0].strip()
                email = parts[1].strip()
    
    generator = HTMLtoPDFGenerator()
    return generator.generate_cover_letter_pdf(
        intro, body, conclusion, name, phone, email, output_path=output_path
    )


def validate_pdf_output(pdf_path: str) -> Dict[str, Any]:
    """
    Validate a generated PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dict with validation results
    """
    result = {
        "exists": False,
        "file_size": 0,
        "is_valid_size": False,
        "estimated_page_count": 0
    }
    
    try:
        if os.path.exists(pdf_path):
            result["exists"] = True
            result["file_size"] = os.path.getsize(pdf_path)
            
            # Rough validation
            if 1000 < result["file_size"] < 10 * 1024 * 1024:
                result["is_valid_size"] = True
                result["estimated_page_count"] = 1
            
    except Exception as e:
        logger.error(f"Error validating PDF: {e}")
    
    return result 