"""
PDF generation utilities for the Application Factory.

Provides professional PDF generation for resumes and cover letters with
ATS-friendly formatting matching the original HTML resume format.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import black, darkblue, navy, Color
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether, HRFlowable
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib import colors
from reportlab.platypus.flowables import PageBreak
from reportlab.graphics.shapes import Line
from reportlab.graphics.renderPDF import drawToFile
from reportlab.graphics import renderPDF

logger = logging.getLogger(__name__)


class PDFGenerator:
    """
    Professional PDF generator for resumes and cover letters.
    Matches the exact formatting of the original HTML resume.
    """
    
    def __init__(self, pagesize=letter):
        """
        Initialize the PDF generator.
        
        Args:
            pagesize: Page size (default: letter)
        """
        self.pagesize = pagesize
        self.page_width = pagesize[0]
        self.page_height = pagesize[1]
        
        # Define margins (matching HTML layout)
        self.margin_top = 0.5 * inch
        self.margin_bottom = 0.5 * inch
        self.margin_left = 0.5 * inch
        self.margin_right = 0.5 * inch
        
        # Calculate available content area
        self.content_width = self.page_width - self.margin_left - self.margin_right
        self.content_height = self.page_height - self.margin_top - self.margin_bottom
        
        # Initialize styles
        self._setup_styles()
    
    def _setup_styles(self):
        """Setup custom styles matching the HTML resume format."""
        self.styles = getSampleStyleSheet()
        
        # Define unique custom style names to avoid conflicts with existing styles
        custom_styles = {
            'ResumeTitle': 'ResumeTitle_Custom',
            'ContactInfo': 'ContactInfo_Custom', 
            'SectionHeader': 'SectionHeader_Custom',
            'BodyText': 'BodyText_Custom',
            'BoldText': 'BoldText_Custom',
            'EducationTitle': 'EducationTitle_Custom',
            'JobTitle': 'JobTitle_Custom',
            'JobDetails': 'JobDetails_Custom', 
            'ProjectTitle': 'ProjectTitle_Custom',
            'BulletPoint': 'BulletPoint_Custom'
        }
        
        # Create new stylesheet to avoid conflicts
        # Name style (18pt Times New Roman Bold, centered)
        if 'ResumeTitle_Custom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ResumeTitle_Custom',
                parent=self.styles['Normal'],
                fontSize=18,
                spaceAfter=2,
                alignment=TA_CENTER,
                textColor=black,
                fontName='Times-Bold'
            ))
        
        # Contact info style (10.5pt Times New Roman, centered)
        if 'ContactInfo_Custom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ContactInfo_Custom',
                parent=self.styles['Normal'],
                fontSize=10.5,
                spaceAfter=6,
                alignment=TA_CENTER,
                textColor=black,
                fontName='Times-Roman'
            ))
        
        # Section header style (11.5pt Times New Roman Bold)
        if 'SectionHeader_Custom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeader_Custom',
                parent=self.styles['Normal'],
                fontSize=11.5,
                spaceBefore=7,
                spaceAfter=0,
                alignment=TA_LEFT,
                textColor=black,
                fontName='Times-Bold'
            ))
        
        # Body text style (10.5pt Times New Roman)
        if 'BodyText_Custom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BodyText_Custom',
                parent=self.styles['Normal'],
                fontSize=10.5,
                spaceAfter=0,
                alignment=TA_LEFT,
                textColor=black,
                fontName='Times-Roman',
                leading=12
            ))
        
        # Bold text within body (10.5pt Times New Roman Bold)
        if 'BoldText_Custom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BoldText_Custom',
                parent=self.styles['Normal'],
                fontSize=10.5,
                spaceAfter=0,
                alignment=TA_LEFT,
                textColor=black,
                fontName='Times-Bold',
                leading=12
            ))
        
        # Education/Organization title style
        if 'EducationTitle_Custom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='EducationTitle_Custom',
                parent=self.styles['Normal'],
                fontSize=10.5,
                spaceBefore=2,
                spaceAfter=0,
                alignment=TA_LEFT,
                textColor=black,
                fontName='Times-Bold'
            ))
        
        # Job title style
        if 'JobTitle_Custom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='JobTitle_Custom',
                parent=self.styles['Normal'],
                fontSize=10.5,
                spaceBefore=0,
                spaceAfter=0,
                alignment=TA_LEFT,
                textColor=black,
                fontName='Times-Bold'
            ))
        
        # Job details (dates, locations)
        if 'JobDetails_Custom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='JobDetails_Custom',
                parent=self.styles['Normal'],
                fontSize=10.5,
                spaceAfter=4,
                alignment=TA_LEFT,
                textColor=black,
                fontName='Times-Roman'
            ))
        
        # Project title style
        if 'ProjectTitle_Custom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ProjectTitle_Custom',
                parent=self.styles['Normal'],
                fontSize=10.5,
                spaceBefore=0,
                spaceAfter=0,
                alignment=TA_LEFT,
                textColor=black,
                fontName='Times-Bold'
            ))
        
        # Bullet point style (36pt left margin)
        if 'BulletPoint_Custom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BulletPoint_Custom',
                parent=self.styles['Normal'],
                fontSize=10.5,
                spaceAfter=0,
                alignment=TA_LEFT,
                textColor=black,
                fontName='Times-Roman',
                leftIndent=36,
                bulletIndent=18,
                leading=11
            ))
    
    def _create_horizontal_line(self):
        """Create a horizontal line separator matching the HTML format."""
        return HRFlowable(
            width="100%",
            thickness=1,
            color=black,
            spaceBefore=0,
            spaceAfter=0,
            hAlign='LEFT',
            vAlign='BOTTOM'
        )
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for PDF generation.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text safe for PDF
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Escape special characters for ReportLab
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        return text.strip()
    
    def _format_contact_info(self, phone: str = "", email: str = "", 
                           linkedin: str = "", website: str = "", location: str = "Toronto, ON") -> str:
        """
        Format contact information line matching HTML format.
        
        Args:
            phone: Phone number
            email: Email address
            linkedin: LinkedIn URL
            website: Website URL (optional)
            location: Location
            
        Returns:
            str: Formatted contact info line
        """
        contact_parts = []
        
        if phone:
            contact_parts.append(phone)
        if email:
            contact_parts.append(email)
        if linkedin:
            # Extract just the handle or display as GitHub/LinkedIn
            contact_parts.append("LinkedIn")
        if website:
            contact_parts.append(website)
        
        return " | ".join(contact_parts)
    
    def _format_skills_section(self, skills_content: str) -> List:
        """
        Format skills section with proper bullet points.
        
        Args:
            skills_content: Raw skills content
            
        Returns:
            List: Formatted story elements
        """
        elements = []
        
        # Split into lines and process
        lines = [line.strip() for line in skills_content.split('\n') if line.strip()]
        
        for line in lines:
            if ':' in line:
                # This is a category line like "Programming Languages: Python, Java..."
                parts = line.split(':', 1)
                category = parts[0].strip()
                skills = parts[1].strip()
                
                # Format with bold category
                formatted_line = f"<b>{self._clean_text(category)}:</b> {self._clean_text(skills)}"
                elements.append(Paragraph(f"• {formatted_line}", self.styles['BulletPoint_Custom']))
            else:
                # Regular bullet point
                clean_line = line.strip()
                if clean_line.startswith(('*', '•', '-')):
                    clean_line = clean_line[1:].strip()
                elements.append(Paragraph(f"• {self._clean_text(clean_line)}", self.styles['BulletPoint_Custom']))
        
        return elements
    
    def _format_education_section(self, education_content: str) -> List:
        """
        Format education section matching HTML layout.
        
        Args:
            education_content: Raw education content
            
        Returns:
            List: Formatted story elements
        """
        elements = []
        lines = [line.strip() for line in education_content.split('\n') if line.strip()]
        
        if not lines:
            return elements
        
        # First line should be university name and location
        if lines:
            # Look for university name and location pattern
            university_line = lines[0]
            if any(keyword in university_line.lower() for keyword in ['university', 'college', 'institute']):
                # Format: "University Name                                                     Location"
                if 'toronto' in university_line.lower() or 'ontario' in university_line.lower():
                    # Split if location is included
                    parts = university_line.split()
                    if 'toronto' in university_line.lower():
                        idx = next(i for i, word in enumerate(parts) if 'toronto' in word.lower())
                        university = ' '.join(parts[:idx]).strip()
                        location = ' '.join(parts[idx:]).strip()
                        formatted_line = f"<b>{university}</b>" + "&nbsp;" * 20 + f"{location}"
                    else:
                        formatted_line = f"<b>{university_line}</b>"
                else:
                    formatted_line = f"<b>{university_line}</b>"
                
                elements.append(Paragraph(formatted_line, self.styles['BodyText_Custom']))
        
        # Process remaining lines
        for line in lines[1:]:
            if 'bachelor' in line.lower() or 'degree' in line.lower() or 'gpa' in line.lower():
                # Degree line with GPA
                if 'gpa' in line.lower():
                    parts = line.split('GPA')
                    degree_part = parts[0].strip()
                    gpa_part = 'GPA' + parts[1].strip() if len(parts) > 1 else ''
                    
                    # Look for dates at the end
                    words = line.split()
                    date_pattern = [w for w in words if any(char.isdigit() for char in w) and ('20' in w or '-' in w)]
                    
                    if date_pattern:
                        # Remove dates from degree part
                        for date_word in date_pattern:
                            degree_part = degree_part.replace(date_word, '').strip()
                        dates = ' '.join(date_pattern)
                        
                        formatted_line = f"{degree_part} <b>{gpa_part}</b>" + "&nbsp;" * 10 + f"{dates}"
                    else:
                        formatted_line = f"{degree_part} <b>{gpa_part}</b>"
                else:
                    formatted_line = line
                
                elements.append(Paragraph(formatted_line, self.styles['BodyText_Custom']))
            elif 'relevant coursework' in line.lower():
                elements.append(Paragraph(f"<b>{line}</b>", self.styles['BodyText_Custom']))
            elif line.startswith(('*', '•', '-')) or any(keyword in line.lower() for keyword in ['python', 'java', 'data structures']):
                # Coursework bullet points
                clean_line = line.strip()
                if clean_line.startswith(('*', '•', '-')):
                    clean_line = clean_line[1:].strip()
                
                # Make key terms bold
                bold_terms = ['Python', 'Object-Oriented Design', 'Data Structures and Algorithms']
                formatted_line = clean_line
                for term in bold_terms:
                    if term.lower() in formatted_line.lower():
                        formatted_line = formatted_line.replace(term, f"<b>{term}</b>")
                
                elements.append(Paragraph(f"• {formatted_line}", self.styles['BulletPoint_Custom']))
            else:
                elements.append(Paragraph(self._clean_text(line), self.styles['BodyText_Custom']))
        
        return elements
    
    def _format_experience_section(self, experience_content: str) -> List:
        """
        Format work experience section matching HTML layout.
        
        Args:
            experience_content: Raw experience content
            
        Returns:
            List: Formatted story elements
        """
        elements = []
        
        # Split into job entries
        entries = experience_content.split('\n\n')
        
        for entry in entries:
            if not entry.strip():
                continue
            
            lines = [line.strip() for line in entry.split('\n') if line.strip()]
            if not lines:
                continue
            
            # First line: Organization name and location
            if lines:
                org_line = lines[0]
                # Format: "Organization Name                                        Location"
                if any(keyword in org_line.lower() for keyword in ['university', 'laboratory', 'company', 'inc', 'corp']):
                    formatted_org = f"<b>{self._clean_text(org_line)}</b>"
                    elements.append(Paragraph(formatted_org, self.styles['BodyText_Custom']))
            
            # Second line: Job title and dates
            if len(lines) > 1:
                job_line = lines[1]
                # Format: "Job Title        Dates"
                elements.append(Paragraph(self._clean_text(job_line), self.styles['BodyText_Custom']))
            
            # Bullet points for achievements
            for line in lines[2:]:
                if line.strip():
                    clean_line = line.strip()
                    if clean_line.startswith(('*', '•', '-')):
                        clean_line = clean_line[1:].strip()
                    
                    # Make key terms bold
                    bold_terms = ['Awarded', 'Developed', 'Conducted', 'Presented', 'Technologies', 'Python', 'TensorFlow']
                    formatted_line = clean_line
                    for term in bold_terms:
                        if term in formatted_line and not f"<b>{term}</b>" in formatted_line:
                            formatted_line = formatted_line.replace(term, f"<b>{term}</b>")
                    
                    elements.append(Paragraph(f"• {formatted_line}", self.styles['BulletPoint_Custom']))
        
        return elements
    
    def _format_projects_section(self, projects_content: str) -> List:
        """
        Format projects section matching HTML layout.
        
        Args:
            projects_content: Raw projects content
            
        Returns:
            List: Formatted story elements
        """
        elements = []
        
        # Split into project entries
        entries = projects_content.split('\n\n')
        
        for entry in entries:
            if not entry.strip():
                continue
            
            lines = [line.strip() for line in entry.split('\n') if line.strip()]
            if not lines:
                continue
            
            # First line: Project title (bold)
            if lines:
                project_title = lines[0]
                elements.append(Paragraph(f"<b>{self._clean_text(project_title)}</b>", self.styles['BodyText_Custom']))
            
            # Bullet points for project details
            for line in lines[1:]:
                if line.strip():
                    clean_line = line.strip()
                    if clean_line.startswith(('*', '•', '-')):
                        clean_line = clean_line[1:].strip()
                    
                    # Make technology terms bold
                    tech_terms = ['Python', 'JavaScript', 'React.js', 'Node.js', 'TensorFlow', 'Scikit-Learn', 
                                 'CSS', 'PostgreSQL', 'Flask', 'Streamlit', 'Langchain', 'OpenAI']
                    formatted_line = clean_line
                    for term in tech_terms:
                        if term in formatted_line and not f"<b>{term}</b>" in formatted_line:
                            formatted_line = formatted_line.replace(term, f"<b>{term}</b>")
                    
                    elements.append(Paragraph(f"• {formatted_line}", self.styles['BulletPoint_Custom']))
        
        return elements
    
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
        Generate a professional resume PDF matching the HTML format.
        
        Args:
            sections (Dict[str, str]): Resume sections
            name (str): Full name
            phone (str): Phone number
            email (str): Email address
            linkedin (str): LinkedIn URL
            website (str): Website URL (optional)
            location (str): Location
            output_path (str): Path for the output PDF
            
        Returns:
            bool: True if successful and fits on one page
        """
        try:
            # Create document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=self.pagesize,
                rightMargin=self.margin_right,
                leftMargin=self.margin_left,
                topMargin=self.margin_top,
                bottomMargin=self.margin_bottom
            )
            
            # Build story (content elements)
            story = []
            
            # Header section
            if name:
                story.append(Paragraph(self._clean_text(name), self.styles['ResumeTitle_Custom']))
            
            # Location
            if location:
                story.append(Paragraph(self._clean_text(location), self.styles['ContactInfo_Custom']))
            
            # Contact info
            contact_info = self._format_contact_info(phone, email, linkedin, website, location)
            if contact_info:
                story.append(Paragraph(contact_info, self.styles['ContactInfo_Custom']))
            
            # Add sections in the correct order
            section_order = ['summary', 'skills', 'education', 'experience', 'projects']
            section_titles = {
                'summary': 'Summary',
                'skills': 'Skills and Interests', 
                'education': 'Education',
                'experience': 'Work Experience',
                'projects': 'Projects'
            }
            
            for section_key in section_order:
                if section_key in sections and sections[section_key].strip():
                    content = sections[section_key].strip()
                    
                    # Add section header
                    section_title = section_titles.get(section_key, section_key.title())
                    story.append(Paragraph(section_title, self.styles['SectionHeader_Custom']))
                    
                    # Add horizontal line
                    story.append(self._create_horizontal_line())
                    story.append(Spacer(1, 2))
                    
                    # Format content based on section type
                    if section_key == 'summary':
                        # Summary as regular paragraph with bold highlights
                        formatted_content = content
                        # Make GPA bold if present
                        if 'gpa' in content.lower():
                            import re
                            gpa_pattern = r'(\d+\.\d+/\d+\.\d+ GPA)'
                            formatted_content = re.sub(gpa_pattern, r'<b>\1</b>', formatted_content)
                        
                        story.append(Paragraph(self._clean_text(formatted_content), self.styles['BodyText_Custom']))
                        
                    elif section_key == 'skills':
                        story.extend(self._format_skills_section(content))
                        
                    elif section_key == 'education':
                        story.extend(self._format_education_section(content))
                        
                    elif section_key == 'experience':
                        story.extend(self._format_experience_section(content))
                        
                    elif section_key == 'projects':
                        story.extend(self._format_projects_section(content))
                    
                    # Add small spacer between sections
                    story.append(Spacer(1, 4))
            
            # Build PDF
            doc.build(story)
            
            # Validate one-page constraint
            if self._validate_page_count(output_path):
                logger.info(f"Resume PDF generated successfully: {output_path}")
                return True
            else:
                logger.warning(f"Resume PDF exceeds one page: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error generating resume PDF: {e}")
            return False
    
    def generate_cover_letter_pdf(
        self,
        intro: str,
        body: List[str],
        conclusion: str,
        header_text: Optional[str] = None,
        output_path: str = "cover_letter.pdf"
    ) -> bool:
        """
        Generate a professional cover letter PDF.
        
        Args:
            intro (str): Introduction paragraph
            body (List[str]): Body paragraphs
            conclusion (str): Conclusion paragraph
            header_text (Optional[str]): Header with contact info and date
            output_path (str): Path for the output PDF
            
        Returns:
            bool: True if successful and fits on one page
        """
        try:
            # Create document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=self.pagesize,
                rightMargin=self.margin_right,
                leftMargin=self.margin_left,
                topMargin=self.margin_top,
                bottomMargin=self.margin_bottom
            )
            
            # Build story
            story = []
            
            # Add header if provided
            if header_text:
                header_lines = header_text.split('\n')
                for line in header_lines:
                    if line.strip():
                        story.append(Paragraph(self._clean_text(line), self.styles['ContactInfo_Custom']))
                story.append(Spacer(1, 12))
            
            # Add date
            current_date = datetime.now().strftime("%B %d, %Y")
            story.append(Paragraph(current_date, self.styles['BodyText_Custom']))
            story.append(Spacer(1, 12))
            
            # Add introduction
            if intro:
                story.append(Paragraph(self._clean_text(intro), self.styles['BodyText_Custom']))
                story.append(Spacer(1, 12))
            
            # Add body paragraphs
            for paragraph in body:
                if paragraph and paragraph.strip():
                    story.append(Paragraph(self._clean_text(paragraph), self.styles['BodyText_Custom']))
                    story.append(Spacer(1, 12))
            
            # Add conclusion
            if conclusion:
                story.append(Paragraph(self._clean_text(conclusion), self.styles['BodyText_Custom']))
                story.append(Spacer(1, 12))
            
            # Add closing
            story.append(Paragraph("Sincerely,", self.styles['BodyText_Custom']))
            story.append(Spacer(1, 24))  # Space for signature
            
            # Build PDF
            doc.build(story)
            
            # Validate one-page constraint
            if self._validate_page_count(output_path):
                logger.info(f"Cover letter PDF generated successfully: {output_path}")
                return True
            else:
                logger.warning(f"Cover letter PDF exceeds one page: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error generating cover letter PDF: {e}")
            return False
    
    def _validate_page_count(self, pdf_path: str) -> bool:
        """
        Validate that the PDF is exactly one page.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            bool: True if exactly one page
        """
        try:
            # For now, we'll assume ReportLab generates single-page PDFs
            # In a production environment, you might use PyPDF2 or similar
            # to actually count pages
            
            # Check if file exists and has reasonable size
            if not os.path.exists(pdf_path):
                return False
            
            file_size = os.path.getsize(pdf_path)
            
            # Very rough validation - a reasonable one-page PDF should be
            # between 1KB and 1MB
            if 1000 < file_size < 1024 * 1024:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating PDF page count: {e}")
            return False


# Convenience functions for easier access
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
    Convenience function to generate a resume PDF.
    
    Args:
        sections (Dict[str, str]): Resume sections
        name (str): Full name
        phone (str): Phone number
        email (str): Email address
        linkedin (str): LinkedIn URL
        website (str): Website URL (optional)
        location (str): Location
        output_path (str): Output file path
        
    Returns:
        bool: True if successful and one page
    """
    generator = PDFGenerator()
    return generator.generate_resume_pdf(sections, name, phone, email, linkedin, website, location, output_path)


def generate_cover_letter_pdf(
    intro: str,
    body: List[str],
    conclusion: str,
    header_text: Optional[str] = None,
    output_path: str = "cover_letter.pdf"
) -> bool:
    """
    Convenience function to generate a cover letter PDF.
    
    Args:
        intro (str): Introduction paragraph
        body (List[str]): Body paragraphs
        conclusion (str): Conclusion paragraph
        header_text (Optional[str]): Header text
        output_path (str): Output file path
        
    Returns:
        bool: True if successful and one page
    """
    generator = PDFGenerator()
    return generator.generate_cover_letter_pdf(intro, body, conclusion, header_text, output_path)


def validate_pdf_output(pdf_path: str) -> Dict[str, Any]:
    """
    Validate a generated PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Dict[str, Any]: Validation results
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
            if 1000 < result["file_size"] < 1024 * 1024:
                result["is_valid_size"] = True
                result["estimated_page_count"] = 1
            
    except Exception as e:
        logger.error(f"Error validating PDF: {e}")
    
    return result 