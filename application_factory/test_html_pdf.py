"""
Test suite for HTML-to-PDF generation functionality.

Tests the HTML template-based PDF generation to ensure it produces
high-quality resumes and cover letters matching the desired format.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pytest

# Add the application_factory directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from html_pdf_utils_alt import HTMLtoPDFGenerator, generate_resume_pdf, generate_cover_letter_pdf, validate_pdf_output

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test data
TEST_RESUME_SECTIONS = {
    "summary": "Results-driven Computer Science student with strong analytical and problem-solving skills. Experienced in software development using Python, Java, and modern web technologies. Passionate about artificial intelligence and machine learning applications.",
    
    "skills": """Programming Languages: Python, Java, JavaScript, TypeScript, C++, SQL
Frameworks & Libraries: React, Node.js, Flask, Django, Spring Boot, TensorFlow
Databases: PostgreSQL, MongoDB, MySQL, Redis
Tools & Technologies: Git, Docker, AWS, Google Cloud Platform, Linux
AI/ML: Machine Learning, Data Analysis, Natural Language Processing""",
    
    "education": """Toronto Metropolitan University (formerly Ryerson University)
Bachelor of Science in Computer Science, GPA: 3.8/4.0
Sep 2023 - Apr 2027

Relevant Coursework:
• Data Structures and Algorithms
• Object-Oriented Programming
• Database Systems
• Software Engineering
• Machine Learning Fundamentals""",
    
    "experience": """Tech Solutions Inc.
Software Development Intern                                    May 2024 - Aug 2024
• Developed and maintained web applications using React and Node.js
• Collaborated with senior developers to implement new features and optimize performance
• Participated in code reviews and followed agile development practices
• Reduced application load time by 25% through code optimization

Local Coffee Shop
Part-time Barista                                              Sep 2023 - Present
• Provided excellent customer service in fast-paced environment
• Managed cash transactions and inventory tracking
• Trained new team members on coffee preparation techniques""",
    
    "projects": """Personal Finance Tracker
• Built a full-stack web application using React frontend and Flask backend
• Implemented user authentication and secure data storage
• Created interactive dashboards for expense visualization and budgeting
• Deployed application using Docker and AWS EC2

AI Chatbot Assistant
• Developed an intelligent chatbot using Python and natural language processing
• Integrated with OpenAI API for enhanced conversational capabilities
• Implemented context awareness and conversation memory
• Achieved 85% user satisfaction in testing with 50+ users"""
}

TEST_CONTACT_INFO = {
    "name": "Gajanan Vigneswaran",
    "phone": "416-555-0123",
    "email": "gajanan.vigneswaran@ryerson.ca",
    "linkedin": "https://linkedin.com/in/gajananv",
    "website": "https://github.com/gajananv",
    "location": "Toronto, ON"
}

TEST_COVER_LETTER_DATA = {
    "introduction": "I am writing to express my strong interest in the Software Developer position at your company. As a Computer Science student with hands-on experience in software development and a passion for creating innovative solutions, I am excited about the opportunity to contribute to your development team.",
    
    "body_paragraphs": [
        "My technical background includes proficiency in Python, Java, and modern web technologies including React and Node.js. During my internship at Tech Solutions Inc., I successfully developed and maintained web applications, collaborated with senior developers, and contributed to improving application performance by 25%. This experience has strengthened my ability to work in agile development environments and deliver high-quality code.",
        
        "What particularly draws me to your company is your commitment to innovation and technology excellence. I am eager to bring my skills in software development, problem-solving abilities, and enthusiasm for learning new technologies to contribute to your team's success. My academic projects, including a personal finance tracker and AI chatbot assistant, demonstrate my ability to build complete solutions from concept to deployment."
    ],
    
    "conclusion": "I would welcome the opportunity to discuss how my technical skills and passion for software development can contribute to your team. Thank you for considering my application. I look forward to hearing from you soon."
}


def test_template_creation():
    """Test that templates exist and are properly formatted."""
    print("\n🧪 Testing template creation...")
    
    generator = HTMLtoPDFGenerator()
    
    # Check if templates directory exists
    templates_dir = Path(__file__).parent / "templates"
    assert templates_dir.exists(), "Templates directory does not exist"
    
    # Check if template files exist
    resume_template = templates_dir / "resume_template.html"
    cover_letter_template = templates_dir / "cover_letter_template.html"
    
    assert resume_template.exists(), "Resume template does not exist"
    assert cover_letter_template.exists(), "Cover letter template does not exist"
    
    print("✅ Templates exist and generator initialized successfully")


def test_resume_generation():
    """Test resume PDF generation."""
    print("\n🧪 Testing resume PDF generation...")
    
    output_path = "test_outputs/test_resume.pdf"
    os.makedirs("test_outputs", exist_ok=True)
    
    # Generate resume
    success = generate_resume_pdf(
        sections=TEST_RESUME_SECTIONS,
        name=TEST_CONTACT_INFO["name"],
        phone=TEST_CONTACT_INFO["phone"],
        email=TEST_CONTACT_INFO["email"],
        linkedin=TEST_CONTACT_INFO["linkedin"],
        website=TEST_CONTACT_INFO["website"],
        location=TEST_CONTACT_INFO["location"],
        output_path=output_path
    )
    
    # Check if PDF was generated successfully
    if success:
        # Validate the PDF output
        validation = validate_pdf_output(output_path)
        
        assert validation["exists"], f"Resume PDF file does not exist: {output_path}"
        assert validation["is_valid_size"], f"Resume PDF has invalid size: {validation}"
        
        print(f"✅ Resume PDF generated successfully")
        print(f"   📄 File: {output_path}")
        print(f"   📊 Size: {validation['file_size']} bytes")
    else:
        # Check if HTML fallback was generated (when wkhtmltopdf is not available)
        html_path = output_path.replace('.pdf', '.html')
        assert os.path.exists(html_path), f"Neither PDF nor HTML fallback was generated"
        
        html_size = os.path.getsize(html_path)
        assert html_size > 1000, f"HTML file too small, likely empty: {html_size} bytes"
        
        print(f"✅ Resume HTML generated successfully (PDF fallback)")
        print(f"   📄 File: {html_path}")
        print(f"   📊 Size: {html_size} bytes")
        print("   ℹ️ Install wkhtmltopdf to generate PDF files")


def test_cover_letter_generation():
    """Test cover letter PDF generation."""
    print("\n🧪 Testing cover letter PDF generation...")
    
    output_path = "test_outputs/test_cover_letter.pdf"
    os.makedirs("test_outputs", exist_ok=True)
    
    # Generate cover letter using class method
    generator = HTMLtoPDFGenerator()
    success = generator.generate_cover_letter_pdf(
        introduction=TEST_COVER_LETTER_DATA["introduction"],
        body_paragraphs=TEST_COVER_LETTER_DATA["body_paragraphs"],
        conclusion=TEST_COVER_LETTER_DATA["conclusion"],
        name=TEST_CONTACT_INFO["name"],
        phone=TEST_CONTACT_INFO["phone"],
        email=TEST_CONTACT_INFO["email"],
        linkedin=TEST_CONTACT_INFO["linkedin"],
        website=TEST_CONTACT_INFO["website"],
        location=TEST_CONTACT_INFO["location"],
        output_path=output_path
    )
    
    # Check if PDF was generated successfully
    if success:
        # Validate the PDF output
        validation = validate_pdf_output(output_path)
        
        assert validation["exists"], f"Cover letter PDF file does not exist: {output_path}"
        assert validation["is_valid_size"], f"Cover letter PDF has invalid size: {validation}"
        
        print(f"✅ Cover letter PDF generated successfully")
        print(f"   📄 File: {output_path}")
        print(f"   📊 Size: {validation['file_size']} bytes")
    else:
        # Check if HTML fallback was generated (when wkhtmltopdf is not available)
        html_path = output_path.replace('.pdf', '.html')
        assert os.path.exists(html_path), f"Neither PDF nor HTML fallback was generated"
        
        html_size = os.path.getsize(html_path)
        assert html_size > 1000, f"HTML file too small, likely empty: {html_size} bytes"
        
        print(f"✅ Cover letter HTML generated successfully (PDF fallback)")
        print(f"   📄 File: {html_path}")
        print(f"   📊 Size: {html_size} bytes")
        print("   ℹ️ Install wkhtmltopdf to generate PDF files")


def test_html_template_rendering():
    """Test HTML template rendering with realistic data."""
    print("\n🧪 Testing HTML template rendering...")
    
    generator = HTMLtoPDFGenerator()
    
    # Test resume template rendering
    resume_context = generator._prepare_resume_context(
        sections=TEST_RESUME_SECTIONS,
        name=TEST_CONTACT_INFO["name"],
        phone=TEST_CONTACT_INFO["phone"],
        email=TEST_CONTACT_INFO["email"],
        linkedin=TEST_CONTACT_INFO["linkedin"],
        website=TEST_CONTACT_INFO["website"],
        location=TEST_CONTACT_INFO["location"]
    )
    
    # Directly use jinja environment to render template
    resume_template = generator.jinja_env.get_template('resume_template.html')
    resume_html = resume_template.render(**resume_context)
    
    assert resume_html is not None, "Resume template rendering failed"
    assert len(resume_html) > 0, "Resume template rendered empty content"
    assert TEST_CONTACT_INFO["name"] in resume_html, "Name not found in rendered resume"
    assert TEST_CONTACT_INFO["email"] in resume_html, "Email not found in rendered resume"
    
    # Test cover letter template rendering
    cover_letter_context = generator._prepare_cover_letter_context(
        introduction=TEST_COVER_LETTER_DATA["introduction"],
        body_paragraphs=TEST_COVER_LETTER_DATA["body_paragraphs"],
        conclusion=TEST_COVER_LETTER_DATA["conclusion"],
        name=TEST_CONTACT_INFO["name"],
        phone=TEST_CONTACT_INFO["phone"],
        email=TEST_CONTACT_INFO["email"],
        linkedin=TEST_CONTACT_INFO["linkedin"],
        website=TEST_CONTACT_INFO["website"],
        location=TEST_CONTACT_INFO["location"]
    )
    
    # Directly use jinja environment to render template
    cover_letter_template = generator.jinja_env.get_template('cover_letter_template.html')
    cover_letter_html = cover_letter_template.render(**cover_letter_context)
    
    assert cover_letter_html is not None, "Cover letter template rendering failed"
    assert len(cover_letter_html) > 0, "Cover letter template rendered empty content"
    assert TEST_CONTACT_INFO["name"] in cover_letter_html, "Name not found in rendered cover letter"
    
    # Save HTML outputs for inspection
    os.makedirs("test_outputs", exist_ok=True)
    with open("test_outputs/test_resume.html", "w", encoding="utf-8") as f:
        f.write(resume_html)
    with open("test_outputs/test_cover_letter.html", "w", encoding="utf-8") as f:
        f.write(cover_letter_html)
    
    print("✅ HTML template rendering successful")
    print("   📄 Resume HTML: test_outputs/test_resume.html")
    print("   📄 Cover Letter HTML: test_outputs/test_cover_letter.html")


def test_dependency_check():
    """Test that required dependencies are available."""
    print("\n🧪 Testing dependency availability...")
    
    # Test jinja2 import
    try:
        import jinja2
        print(f"✅ Jinja2 version: {jinja2.__version__}")
    except ImportError as e:
        pytest.fail(f"Jinja2 not available: {e}")
    
    # Test weasyprint import (optional)
    try:
        import weasyprint
        print(f"✅ WeasyPrint version: {weasyprint.__version__}")
        weasyprint_available = True
    except ImportError:
        print("⚠️ WeasyPrint not available (using fallback)")
        weasyprint_available = False
    
    # Test pdfkit import (alternative)
    try:
        import pdfkit
        print("✅ pdfkit available")
        pdfkit_available = True
    except ImportError:
        print("⚠️ pdfkit not available")
        pdfkit_available = False
    
    # At least one PDF generation method should be available
    assert weasyprint_available or pdfkit_available, "No PDF generation method available"
    
    print("✅ Dependencies check passed") 