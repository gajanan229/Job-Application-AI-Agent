# 🏭 Application Factory - AI-Powered Document Crafting Studio

Your personal AI assistant for creating tailored resumes and cover letters using advanced RAG (Retrieval-Augmented Generation) technology.

## 🌟 Overview

The Application Factory is an intelligent document processing system that takes your master resume and job descriptions to generate perfectly tailored application materials. Using Google's Gemini AI, LangGraph workflows, and advanced RAG pipelines, it creates personalized resumes and cover letters that match specific job requirements.

## ✨ Features

### Phase 1: Foundation ✅ COMPLETED
- **Robust Configuration System**: Environment-based settings with validation
- **Advanced Logging**: Structured logging with rotation and performance tracking
- **Session Management**: Streamlit session state management with error handling
- **File Operations**: Secure file handling with validation and cleanup
- **Error Handling**: Comprehensive error management with user-friendly messages

### Phase 2: RAG Pipeline ✅ COMPLETED
- **Multi-Engine PDF Processing**: Robust text extraction using pdfplumber and PyPDF2
- **Intelligent Text Chunking**: Smart document segmentation with metadata preservation
- **Vector Embeddings**: Google Gemini embeddings with FAISS vector storage
- **Semantic Search**: AI-powered context retrieval across documents
- **Document Analysis**: Comprehensive content analysis and statistics

### Coming Soon:
- **Phase 3**: LangGraph State Management
- **Phase 4**: Gemini LLM Integration  
- **Phase 5**: Prompt Engineering
- **Phase 6**: Resume Generation
- **Phase 7**: Cover Letter Generation
- **Phase 8**: Document Templates
- **Phase 9**: DOCX to PDF Conversion
- **Phase 10**: Testing & Deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google API Key (for Gemini embeddings)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd CV-AI-Agent
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
Create a `.env` file in the project root:
```env
# Google AI Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Application Settings
APP_ENV=development
DEBUG=true
MAX_FILE_SIZE_MB=50

# RAG Processing Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=5

# LLM Settings (for future phases)
GEMINI_MODEL=gemini-1.5-flash
TEMPERATURE=0.7
MAX_TOKENS=4000
```

5. **Run the application**:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
application_factory/
├── app.py                     # Main Streamlit application
├── config/
│   ├── __init__.py
│   ├── settings.py            # Configuration management
│   └── logging_config.py      # Logging setup
├── core/
│   ├── __init__.py
│   └── rag_processor.py       # RAG pipeline components
├── utils/
│   ├── __init__.py
│   ├── file_utils.py          # File operations
│   ├── session_utils.py       # Session management
│   └── error_handlers.py      # Error handling
├── tests/
│   ├── __init__.py
│   ├── test_config.py         # Configuration tests
│   └── test_rag_processor.py  # RAG pipeline tests
├── generated_documents/
│   └── temp/                  # Temporary file storage
├── Example CV/                # Template documents
│   ├── CoverLetter_GajananV_AAD.docx
│   ├── Gajanan_Vig_Resume_AAD (1).docx
│   ├── cover_letter_template.docx
│   └── resume_template.docx
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── README.md                 # This file
└── PHASE_2_SUMMARY.md        # Phase 2 completion summary
```

## 🔧 Configuration

### Environment Variables

The application uses environment variables for configuration. Copy `.env.example` to `.env` and configure:

```env
# Required
GOOGLE_API_KEY=your_api_key_here

# Optional (with defaults)
APP_ENV=development
DEBUG=true
MAX_FILE_SIZE_MB=50
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=5
GEMINI_MODEL=gemini-1.5-flash
TEMPERATURE=0.7
MAX_TOKENS=4000
```

### Getting a Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

## 📱 User Guide

### Step 1: Document Upload
1. **Configure API Key**: Enter your Google API key in the sidebar
2. **Upload Master Resume**: Upload your comprehensive resume PDF
3. **Upload Job Description**: Upload the target job description PDF
4. **Validation**: Both files are automatically validated

### Step 2: Document Processing
1. **Start Processing**: Click "Start RAG Processing" when both documents are uploaded
2. **Progress Tracking**: Watch real-time progress as documents are processed
3. **AI Analysis**: Documents are chunked, embedded, and indexed for semantic search

### Step 3: Results and Analysis
1. **Processing Statistics**: View document analysis and chunking results
2. **Test Search**: Try semantic search across your documents
3. **Content Preview**: Explore document chunks and embeddings
4. **Ready for Generation**: Processed documents ready for resume/cover letter creation

## 🏗️ Architecture

### RAG Pipeline Components

1. **PDFProcessor**: Multi-engine PDF text extraction
   - Primary: pdfplumber (better for complex layouts)
   - Fallback: PyPDF2 (broader compatibility)
   - Content analysis and statistics

2. **TextChunker**: Intelligent document segmentation
   - Recursive character text splitting
   - Configurable chunk size and overlap
   - Metadata preservation

3. **EmbeddingManager**: Vector embedding creation
   - Google Gemini embeddings integration
   - Batch processing for efficiency
   - Error handling and validation

4. **VectorStore**: Similarity search engine
   - FAISS vector database
   - L2 distance similarity scoring
   - Configurable result ranking

5. **RAGProcessor**: Main orchestrator
   - End-to-end pipeline coordination
   - Multi-document processing
   - Context retrieval and search

### Session Management
- Streamlit session state management
- Document processing state preservation
- Error and success message handling
- Multi-stage workflow navigation

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_rag_processor.py -v

# Run with coverage
python -m pytest tests/ --cov=core --cov=utils --cov=config
```

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Error Handling**: Exception and edge case testing
- **Mock Testing**: API call simulation

## 📊 Performance

### Processing Metrics
- **PDF Processing**: 2-5 seconds per document
- **Text Chunking**: 500-2000 chunks per document
- **Embedding Creation**: 1-3 seconds per batch
- **Vector Search**: Sub-second response times
- **File Size Limit**: Up to 50MB per document

### System Requirements
- **Memory**: 2GB RAM minimum (4GB recommended)
- **Storage**: 1GB free space for temporary files
- **Network**: Internet connection for Google API calls
- **Browser**: Modern web browser for Streamlit interface

## 🔒 Security and Privacy

### Data Handling
- **Local Processing**: Documents processed locally
- **API Calls**: Only text chunks sent to Google for embeddings
- **Temporary Storage**: Files automatically cleaned up
- **No Persistence**: Document content not permanently stored
- **Session Isolation**: Each user session independent

### API Security
- **Environment Variables**: API keys stored securely in .env
- **No Hardcoding**: Credentials never committed to code
- **Validation**: API key validation before processing
- **Error Handling**: Secure error messages without key exposure

## 🚨 Troubleshooting

### Common Issues

1. **API Key Invalid**:
   - Verify your Google API key is correct
   - Check if the key has necessary permissions
   - Ensure no extra spaces in the .env file

2. **PDF Processing Fails**:
   - Check if PDF is corrupted or image-only
   - Try a different PDF file
   - Ensure file size is under 50MB

3. **Streamlit Errors**:
   - Restart the application
   - Clear browser cache
   - Check Python version compatibility

4. **Memory Issues**:
   - Reduce document size
   - Adjust chunk size in configuration
   - Close other applications

### Debug Mode
Enable debug mode in `.env`:
```env
DEBUG=true
```

This provides:
- Detailed error messages
- Processing logs
- Session state information
- Performance metrics

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests before committing
5. Submit a pull request

### Code Standards
- **PEP 8**: Python code style
- **Type Hints**: Use type annotations
- **Docstrings**: Document all functions
- **Error Handling**: Comprehensive exception handling
- **Testing**: Write tests for new features

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: LLM framework and document processing
- **Google AI**: Gemini embeddings and language models
- **FAISS**: High-performance similarity search
- **Streamlit**: Web application framework
- **pdfplumber**: PDF text extraction
- **PyPDF2**: PDF processing library

## 📞 Support

For issues, questions, or contributions:
- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Documentation**: Check project wiki
- **Updates**: Follow repository for updates

---

**Application Factory** - Transforming job applications with AI-powered document crafting.

*Current Version: Phase 2 RAG Pipeline - COMPLETED ✅* 