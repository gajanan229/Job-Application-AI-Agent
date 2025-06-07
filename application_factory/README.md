# 🏭 Application Factory

**Craft Perfect Resumes and Cover Letters with AI**

The Application Factory is an intelligent document generation system that transforms your Master Resume and a specific Job Description into perfectly tailored, one-page Resume and Cover Letter PDFs using RAG-enhanced AI.

## 🌟 Features

- **RAG-Enhanced AI**: Uses vector search to ground AI generation in your actual experience
- **Intelligent Tailoring**: Automatically adapts content to match job requirements
- **One-Page Guarantee**: Strict enforcement of single-page documents
- **Iterative Refinement**: Feedback loops for perfect content creation
- **Secure API Handling**: Multiple secure methods for API key management
- **Professional PDFs**: High-quality, formatted documents ready for submission
- **Organized Output**: Automatic job-specific folder creation and file management

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone and Setup**
   ```bash
   cd application_factory
   pip install -r requirements.txt
   ```

2. **Configure API Key** (Choose one method)

   **Option A: Streamlit Secrets (Recommended)**
   ```bash
   mkdir -p .streamlit
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # Edit .streamlit/secrets.toml with your API key
   ```

   **Option B: Environment Variable**
   ```bash
   cp .env.example .env
   # Edit .env with your API key
   export GEMINI_API_KEY=your_api_key_here
   ```

   **Option C: Direct Input**
   - Enter your API key directly in the Streamlit interface

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
application_factory/
├── app.py                     # Main Streamlit application
├── graph_builder.py           # LangGraph workflow definition
├── nodes_rag.py              # LangGraph node implementations
├── prompts_gemini.py          # AI prompts for Gemini
├── state_rag.py              # State management
├── rag_utils.py              # RAG utilities (chunking, embeddings, vector store)
├── pdf_utils.py              # PDF generation utilities
├── config/                    # Configuration modules
│   ├── __init__.py
│   ├── api_keys.py           # Secure API key management
│   ├── settings.py           # Application settings
│   ├── paths.py              # Path management utilities
│   └── validators.py         # Input validation
├── assets/                    # Static assets
├── data/                      # Data storage
│   └── vector_store/         # Vector store persistence
└── generated_applications/   # Output directory
```

## 🎯 How It Works

### Stage 1: Resume Builder
1. **Upload Master Resume**: Your comprehensive resume file (.txt or .md)
2. **Enter Job Description**: The specific job you're targeting
3. **AI Analysis**: RAG-enhanced analysis extracts relevant experience
4. **Section-by-Section Generation**: AI crafts each resume section
5. **Interactive Editing**: Review and refine each section
6. **PDF Preview**: Real-time preview with one-page validation
7. **Finalization**: Save polished, job-specific resume

### Stage 2: Cover Letter Builder
1. **Intelligent Context**: Uses your tailored resume and job description
2. **Intro & Conclusion**: AI generates compelling opening and closing
3. **Iterative Refinement**: Feedback loops for perfect tone
4. **Body Paragraphs**: Narrative connecting your experience to job needs
5. **Version Management**: Compare different versions and rollback
6. **PDF Generation**: Professional cover letter ready for submission

### Stage 3: Organization
- Automatic job-specific folder creation
- Consistent file naming conventions
- Ready for submission or future automated tools

## 🔧 Configuration

### Application Settings
Edit `config/settings.py` to customize:
- File size limits
- Text processing parameters
- RAG configuration
- PDF formatting
- UI preferences

### Security
- API keys are **never** stored in application state
- Multiple secure key sources with priority fallback
- Input validation and sanitization
- Safe path handling

## 📊 Usage Tips

### Master Resume Best Practices
- Include comprehensive work experience
- List all skills and technologies
- Detail all projects with technologies used
- Use bullet points for better parsing
- Include education and certifications

### Job Description Input
- Copy the complete job posting
- Include company information if available
- Ensure requirements and responsibilities are clear

### Optimization
- The system works best with detailed master resumes
- More specific job descriptions yield better results
- Review and edit AI-generated content for accuracy
- Use the iterative refinement features

## 🚧 Current Status: Phase 1 Complete

✅ **Phase 1: Foundation & Core Infrastructure**
- [x] Project structure and dependencies
- [x] Security and configuration management
- [x] Core state management
- [x] Input validation system

🔄 **Next Phases**
- Phase 2: RAG Infrastructure & Utilities
- Phase 3: PDF Generation System
- Phase 4: LLM Integration & Prompt Engineering
- Phase 5: LangGraph Workflow Implementation
- Phase 6: Streamlit User Interface
- Phase 7: Integration & Testing
- Phase 8: Polish & Production Readiness

## 🛠️ Development

### Testing
```bash
# Run tests (when implemented)
pytest

# Code formatting
black .
flake8 .
```

### Contributing
1. Follow the detailed implementation plan in phases
2. Maintain security best practices
3. Add comprehensive tests for new features
4. Update documentation

## 📝 License

This project is part of a larger AI-powered job application system. See the main project documentation for licensing information.

## 🤝 Support

For issues, questions, or contributions, please refer to the main project repository.

---

**Note**: This is currently a temporary Streamlit interface. Future versions will include automated job portal submission and integration with the broader job application ecosystem. 