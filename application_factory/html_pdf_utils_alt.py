"""
Alternative HTML-to-PDF generation utilities using pdfkit for Windows compatibility.

Provides professional PDF generation for resumes and cover letters using
HTML templates and pdfkit for conversion, maintaining exact formatting
from the provided example HTML files.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

logger = logging.getLogger(__name__)


class HTMLtoPDFGenerator:
    """
    Professional PDF generator using HTML templates and pdfkit.
    Windows-compatible alternative to WeasyPrint.
    """
    
    def __init__(self, templates_dir: str = None):
        """
        Initialize the HTML-to-PDF generator.
        
        Args:
            templates_dir: Directory containing HTML templates
        """
        if not PDFKIT_AVAILABLE:
            raise ImportError("pdfkit not available. Install with: pip install pdfkit")
        
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
        
        # PDF options for better quality
        self.pdf_options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        
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
        
        # Add sections - simplified formatting for pdfkit compatibility
        for section_name, content in sections.items():
            if content and content.strip():
                context[section_name] = self._clean_text(content)
        
        return context
    
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
            
            # Try to generate PDF with pdfkit
            try:
                pdfkit.from_string(html_content, output_path, options=self.pdf_options)
            except OSError as e:
                if "wkhtmltopdf" in str(e).lower():
                    # wkhtmltopdf not found, fall back to simple HTML output with instructions
                    html_path = output_path.replace('.pdf', '.html')
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    logger.warning(f"⚠️ wkhtmltopdf not found. Generated HTML instead: {html_path}")
                    logger.info("To generate PDF: Install wkhtmltopdf from https://wkhtmltopdf.org/downloads.html")
                    return False
                else:
                    raise
            
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
            
            # Try to generate PDF with pdfkit
            try:
                pdfkit.from_string(html_content, output_path, options=self.pdf_options)
            except OSError as e:
                if "wkhtmltopdf" in str(e).lower():
                    # wkhtmltopdf not found, fall back to simple HTML output with instructions
                    html_path = output_path.replace('.pdf', '.html')
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    logger.warning(f"⚠️ wkhtmltopdf not found. Generated HTML instead: {html_path}")
                    logger.info("To generate PDF: Install wkhtmltopdf from https://wkhtmltopdf.org/downloads.html")
                    return False
                else:
                    raise
            
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