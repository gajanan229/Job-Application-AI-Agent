"""
Content validation utilities for the Application Factory.

This module provides functions to validate generated content length,
estimate page limits, and ensure documents fit within constraints.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ContentMetrics:
    """Metrics for content analysis."""
    word_count: int
    character_count: int
    line_count: int
    estimated_pages: float
    fits_one_page: bool


@dataclass
class ValidationResult:
    """Result of content validation."""
    is_valid: bool
    metrics: ContentMetrics
    issues: List[str]
    suggestions: List[str]


class ContentValidator:
    """Validates content length and page constraints."""
    
    # Page constraints (based on standard business letter formatting)
    WORDS_PER_PAGE_RESUME = 450  # Dense formatting with bullets
    WORDS_PER_PAGE_COVER_LETTER = 350  # Standard paragraph formatting
    CHARS_PER_LINE = 80
    LINES_PER_PAGE = 45
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.ContentValidator")
    
    def analyze_content(self, content: str, content_type: str = "general") -> ContentMetrics:
        """
        Analyze content and calculate metrics.
        
        Args:
            content: Text content to analyze
            content_type: Type of content ("resume", "cover_letter", "general")
            
        Returns:
            ContentMetrics with analysis results
        """
        # Basic metrics
        word_count = len(content.split())
        character_count = len(content)
        line_count = content.count('\n') + 1
        
        # Estimate pages based on content type
        if content_type.lower() == "resume":
            estimated_pages = word_count / self.WORDS_PER_PAGE_RESUME
        elif content_type.lower() == "cover_letter":
            estimated_pages = word_count / self.WORDS_PER_PAGE_COVER_LETTER
        else:
            # General estimation
            estimated_pages = max(
                word_count / 400,  # Conservative word-based estimate
                line_count / self.LINES_PER_PAGE  # Line-based estimate
            )
        
        fits_one_page = estimated_pages <= 1.0
        
        return ContentMetrics(
            word_count=word_count,
            character_count=character_count,
            line_count=line_count,
            estimated_pages=estimated_pages,
            fits_one_page=fits_one_page
        )
    
    def validate_resume_content(self, content_dict: Dict[str, str]) -> ValidationResult:
        """
        Validate complete resume content.
        
        Args:
            content_dict: Dictionary with resume sections
            
        Returns:
            ValidationResult with validation outcome
        """
        issues = []
        suggestions = []
        
        # Combine all content
        all_content = []
        if content_dict.get('summary'):
            all_content.append(content_dict['summary'])
        if content_dict.get('project_bullets'):
            all_content.append(content_dict['project_bullets'])
        
        combined_content = '\n'.join(all_content)
        metrics = self.analyze_content(combined_content, "resume")
        
        # Validate summary length
        if content_dict.get('summary'):
            summary_words = len(content_dict['summary'].split())
            if summary_words > 60:
                issues.append(f"Resume summary too long ({summary_words} words, should be ≤60)")
                suggestions.append("Shorten summary to 2-3 concise sentences")
        
        # Validate projects section
        if content_dict.get('project_bullets'):
            project_content = content_dict['project_bullets']
            project_count = len(re.findall(r'^[A-Z].*(?:\n●.*)*', project_content, re.MULTILINE))
            
            if project_count > 5:
                issues.append(f"Too many projects ({project_count}, should be ≤5)")
                suggestions.append("Limit to top 3-5 most relevant projects")
            
            if metrics.estimated_pages > 1.0:
                issues.append(f"Resume content too long ({metrics.estimated_pages:.1f} pages)")
                suggestions.append("Reduce project bullet points or remove less relevant projects")
        
        # Overall page validation
        if not metrics.fits_one_page:
            issues.append(f"Content exceeds one page ({metrics.estimated_pages:.1f} pages)")
            suggestions.append("Reduce content to fit within one page limit")
        
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            metrics=metrics,
            issues=issues,
            suggestions=suggestions
        )
    
    def validate_cover_letter_content(self, content_dict: Dict[str, str]) -> ValidationResult:
        """
        Validate complete cover letter content.
        
        Args:
            content_dict: Dictionary with cover letter sections
            
        Returns:
            ValidationResult with validation outcome
        """
        issues = []
        suggestions = []
        
        # Combine all content
        all_content = []
        for section in ['introduction', 'body', 'conclusion']:
            if content_dict.get(section):
                all_content.append(content_dict[section])
        
        combined_content = '\n\n'.join(all_content)
        metrics = self.analyze_content(combined_content, "cover_letter")
        
        # Validate individual sections
        if content_dict.get('introduction'):
            intro_words = len(content_dict['introduction'].split())
            if intro_words > 80:
                issues.append(f"Introduction too long ({intro_words} words, should be ≤80)")
                suggestions.append("Shorten introduction to 3-4 sentences")
        
        if content_dict.get('body'):
            body_words = len(content_dict['body'].split())
            if body_words > 160:
                issues.append(f"Body too long ({body_words} words, should be ≤160)")
                suggestions.append("Limit body to 2 concise paragraphs (60-80 words each)")
        
        if content_dict.get('conclusion'):
            conclusion_words = len(content_dict['conclusion'].split())
            if conclusion_words > 60:
                issues.append(f"Conclusion too long ({conclusion_words} words, should be ≤60)")
                suggestions.append("Shorten conclusion to 2-3 sentences")
        
        # Overall page validation
        if not metrics.fits_one_page:
            issues.append(f"Cover letter too long ({metrics.estimated_pages:.1f} pages)")
            suggestions.append("Reduce content to fit within one page limit")
        
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            metrics=metrics,
            issues=issues,
            suggestions=suggestions
        )
    
    def truncate_content_to_fit(self, content: str, max_words: int) -> str:
        """
        Truncate content to fit within word limit while preserving structure.
        
        Args:
            content: Content to truncate
            max_words: Maximum word count
            
        Returns:
            Truncated content
        """
        words = content.split()
        if len(words) <= max_words:
            return content
        
        # Try to preserve sentence structure
        truncated_words = words[:max_words]
        truncated_text = ' '.join(truncated_words)
        
        # Try to end at a sentence boundary
        last_period = truncated_text.rfind('.')
        if last_period > len(truncated_text) * 0.8:  # If period is in last 20%
            truncated_text = truncated_text[:last_period + 1]
        
        self.logger.warning(f"Content truncated from {len(words)} to {len(truncated_text.split())} words")
        return truncated_text
    
    def get_content_summary(self, content: str, content_type: str = "general") -> str:
        """Get a summary string of content metrics."""
        metrics = self.analyze_content(content, content_type)
        
        return (f"Words: {metrics.word_count}, "
                f"Pages: {metrics.estimated_pages:.1f}, "
                f"Fits 1 page: {'✅' if metrics.fits_one_page else '❌'}")


# Convenience functions
def validate_resume(content_dict: Dict[str, str]) -> ValidationResult:
    """Validate resume content."""
    validator = ContentValidator()
    return validator.validate_resume_content(content_dict)


def validate_cover_letter(content_dict: Dict[str, str]) -> ValidationResult:
    """Validate cover letter content."""
    validator = ContentValidator()
    return validator.validate_cover_letter_content(content_dict)


def get_content_metrics(content: str, content_type: str = "general") -> ContentMetrics:
    """Get content metrics."""
    validator = ContentValidator()
    return validator.analyze_content(content, content_type) 