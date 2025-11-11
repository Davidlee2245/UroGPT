# UroGPT - AI-Powered Urinalysis Interpretation System

<div align="center">

![UroGPT](https://img.shields.io/badge/UroGPT-v1.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An intelligent medical AI system that combines computer vision and natural language processing to analyze urinalysis results and generate comprehensive medical reports.**

</div>

---

## 🎯 Overview

UroGPT is a modular AI pipeline designed for urinalysis interpretation, consisting of:

1. **UroAI (Image Analysis Module)**: Analyzes urinalysis strip images to extract quantitative parameters
2. **UroGPT (LLM Agent)**: Interprets results using advanced language models with medical knowledge retrieval (RAG)

### Key Features

- 🔬 **Automated Urinalysis**: Extract glucose, pH, nitrite, and lymphocyte levels from test strips
- 🧠 **AI-Powered Interpretation**: Natural language medical reports using GPT-4/Claude/Gemini
- 📚 **Knowledge Retrieval (RAG)**: Evidence-based interpretations using medical literature
- 🚨 **UTI Detection**: Probabilistic assessment of urinary tract infections
- 🌐 **REST API**: Easy integration via FastAPI endpoints
- 📊 **Structured Output**: JSON format with detailed parameter analysis

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      UroGPT System                       │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│   UroAI (Dummy)     │       │   UroGPT (LLM)      │
│  Image Analysis     │       │   RAG Pipeline      │
└─────────────────────┘       └─────────────────────┘
          │                               │
          ▼                               ▼
   Urinalysis Results ──────────▶  Medical Report
```

### System Components

```
uroai_project/
├── main.py                    # Entry point (CLI + server)
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation
│
├── api/                       # FastAPI endpoints
│   ├── __init__.py
│   └── app.py                 # REST API server
│
├── image_analysis/            # UroAI module (DUMMY)
│   ├── __init__.py
│   └── analyzer.py            # ImageAnalyzer, ExpertModel, AttentionFusion
│
├── llm_agent/                 # UroGPT module (FUNCTIONAL)
│   ├── __init__.py
│   ├── rag_pipeline.py        # RAG with LangChain + FAISS
│   └── generator.py           # LLM report generation
│
└── documents/                 # Medical knowledge corpus
    └── sample_docs/
        ├── urinalysis_basics.txt
        └── uti_management.txt
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8 or higher** (for backend)
- **Node.js 18+** (for React frontend)
- **OpenAI API key** (for GPT-4)
- **4GB+ RAM** recommended

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/Davidlee2245/UroGPT.git
cd UroGPT
```

#### 2. Set up Python Backend
```bash
# Create conda environment (recommended)
conda create -n urogpt python=3.12
conda activate urogpt

# Install Python dependencies
pip install -r requirements.txt
```

#### 3. Set up React Frontend
```bash
cd urogpt-ui
npm install
cd ..
```

#### 4. Configure environment variables
```bash
export OPENAI_API_KEY="your-openai-api-key-here"

# Optional configurations
export LLM_MODEL="gpt-4"                    # or gpt-3.5-turbo
export EMBEDDING_MODEL="openai"             # or huggingface
export CORPUS_PATH="documents/sample_docs"
export PORT="8000"
```

Or create a `.env` file:
```env
OPENAI_API_KEY=your-openai-api-key-here
LLM_MODEL=gpt-4
EMBEDDING_MODEL=openai
CORPUS_PATH=documents/sample_docs
PORT=8000
```

---

## 💻 Usage

### 1. API Server Mode (Recommended)

Start the FastAPI server:

```bash
python main.py --mode api
```

The server will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

#### API Endpoints

**POST /chat** - Chat with AI assistant
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_context": "What does positive nitrite indicate?"
  }'
```

**POST /analyze** - Analyze manual results
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "glucose": 3.1,
    "pH": 6.8,
    "nitrite": 0.2,
    "lymphocyte": 1.4,
    "patient_context": "Optional patient history"
  }'
```

**GET /documents** - List all medical documents
```bash
curl "http://localhost:8000/documents"
```

**GET /documents/content** - Get document content
```bash
curl "http://localhost:8000/documents/content?filepath=documents/sample_docs/urinalysis_basics.txt"
```

**GET /health** - Check system status
```bash
curl "http://localhost:8000/health"
```

### 2. Interactive Mode

Run interactive terminal interface:

```bash
python main.py --mode interactive
```

Enter urinalysis parameters manually and receive instant analysis.

### 3. Single Image Analysis

Analyze a specific image:

```bash
python main.py --mode analyze --image path/to/urinalysis_strip.jpg
```

---

## 📊 Example Output

### Input (JSON)
```json
{
  "glucose": 3.1,
  "pH": 6.8,
  "nitrite": 0.2,
  "lymphocyte": 1.4,
  "UTI_probability": 0.86
}
```

### Output (Medical Report)

```
Urinalysis Report
==================

PARAMETER ANALYSIS:

Glucose: 3.1 mg/dL
✓ Within normal range (0-15 mg/dL). No indication of glucosuria or diabetes.

pH: 6.8
✓ Normal acidic urine (4.5-8.0). Appropriate for kidney function assessment.

Nitrite: 0.2 mg/dL
⚠️ POSITIVE - Indicates presence of gram-negative bacteria. Common in urinary 
tract infections caused by E. coli, Proteus, or Klebsiella species.

Lymphocytes: 1.4 cells/μL
✓ Within normal range (<5 cells/μL). Mild elevation but not clinically significant.

UTI ASSESSMENT:

AI-Predicted Probability: 86%
Risk Level: HIGH

The combination of positive nitrite with other indicators suggests a strong 
likelihood of urinary tract infection. While lymphocyte count is not markedly 
elevated, the positive nitrite test is highly specific for bacteriuria.

CLINICAL RECOMMENDATIONS:

1. Consult healthcare provider for proper diagnosis and treatment
2. Urine culture recommended to identify causative organism
3. Consider empiric antibiotic therapy pending culture results
4. Increase fluid intake
5. Monitor symptoms (dysuria, frequency, urgency, fever)
6. Follow up if symptoms persist or worsen

This report is AI-generated and should not replace professional medical evaluation.
```

---

## 🧪 Testing the System

### Test with sample data:

```python
from image_analysis import ImageAnalyzer
from llm_agent import RAGPipeline, ReportGenerator

# Initialize components
analyzer = ImageAnalyzer()
rag = RAGPipeline()
rag.build_vector_store()
generator = ReportGenerator(rag_pipeline=rag)

# Analyze (dummy results)
results = analyzer.analyze("dummy_image.jpg")

# Generate report
report = generator.generate_report(results)
print(report["report"])
```

---

## 🔧 Configuration

### LLM Models

UroGPT supports multiple language models:

**OpenAI (Default)**
```bash
export LLM_MODEL="gpt-4"  # or gpt-3.5-turbo, gpt-4-turbo
export OPENAI_API_KEY="your-key"
```

**Anthropic Claude**
```python
# Install: pip install anthropic
export LLM_MODEL="claude-3-opus-20240229"
export ANTHROPIC_API_KEY="your-key"
```

**Google Gemini**
```python
# Install: pip install google-generativeai
export LLM_MODEL="gemini-pro"
export GOOGLE_API_KEY="your-key"
```

### Embedding Models

**OpenAI Embeddings (Default)**
```bash
export EMBEDDING_MODEL="openai"
```

**HuggingFace (Free, Local)**
```bash
export EMBEDDING_MODEL="huggingface"
# Uses: sentence-transformers/all-MiniLM-L6-v2
```

---

## 📚 RAG Knowledge Base

The system uses Retrieval-Augmented Generation (RAG) to ground responses in medical evidence.

### Adding Medical Documents

1. Place text files in `documents/sample_docs/`
2. Restart the system to rebuild the vector store
3. Documents will be automatically embedded and indexed

```bash
documents/sample_docs/
├── urinalysis_basics.txt      # Urinalysis parameters
├── uti_management.txt         # UTI diagnosis and treatment
└── your_custom_docs.txt       # Add your own medical references
```

---

## ⚠️ Important Notes

### Current Status

- **Image Analysis Module (UroAI)**: ⚠️ **DUMMY IMPLEMENTATION**
  - Returns placeholder values
  - Real CV model integration pending
  - All classes are stubs with `pass` statements
  
- **LLM Agent (UroGPT)**: ✅ **FULLY FUNCTIONAL**
  - Complete RAG pipeline with LangChain
  - Real LLM integration (OpenAI/Claude/Gemini)
  - Production-ready medical report generation

### Medical Disclaimer

**This system is for research and educational purposes only.**

- NOT FDA approved
- NOT a substitute for professional medical diagnosis
- AI predictions should be verified by qualified healthcare professionals
- Always consult a licensed physician for medical decisions

---

## 🛠️ Development

### Project Structure

- **image_analysis/**: Placeholder for computer vision model
  - TODO: Integrate trained PyTorch model
  - TODO: Implement preprocessing pipeline
  - TODO: Add attention-based fusion mechanism

- **llm_agent/**: Production-ready NLP pipeline
  - RAG with FAISS vector store
  - LangChain integration
  - Multi-model LLM support

- **api/**: REST API with FastAPI
  - OpenAPI documentation
  - File upload support
  - CORS enabled

### Future Enhancements

- [ ] Replace dummy image analyzer with real trained model
- [ ] Add support for more urinalysis parameters
- [ ] Implement user authentication
- [ ] Add result history and tracking
- [ ] Create web-based UI (React/Vue)
- [ ] Add multi-language support
- [ ] Integrate with EHR systems (HL7/FHIR)
- [ ] Add batch processing capabilities

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

1. **Computer Vision**: Help integrate a real urinalysis image analysis model
2. **Medical Knowledge**: Add more medical documents to the corpus
3. **LLM Integration**: Add support for more language models
4. **Testing**: Write unit and integration tests
5. **Documentation**: Improve guides and examples

---

## 📝 License

MIT License - see LICENSE file for details

---

## 📧 Contact

For questions, issues, or collaboration:

- **GitHub Issues**: https://github.com/yourusername/UroGPT/issues
- **Email**: your.email@example.com

---

## 🙏 Acknowledgments

- OpenAI for GPT models
- LangChain for RAG framework
- FastAPI for web framework
- Medical professionals who provided domain expertise

---

**Made with ❤️ for advancing medical AI**

