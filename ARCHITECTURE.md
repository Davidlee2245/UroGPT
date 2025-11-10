# UroGPT System Architecture

## 🏗️ High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                         UroGPT System                              │
│                   AI-Powered Urinalysis Platform                   │
└───────────────────────────────────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
┌───────────────────────────┐       ┌──────────────────────────────┐
│    UroAI (Image Module)   │       │   UroGPT (LLM Module)        │
│        ⚠️ DUMMY           │       │      ✅ FUNCTIONAL           │
└───────────────────────────┘       └──────────────────────────────┘
                │                                 │
                ▼                                 ▼
        Urinalysis Results ──────────▶   Medical Report
         (JSON Format)                 (Natural Language)
```

## 📊 Detailed Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Interfaces                             │
├──────────────┬──────────────────┬──────────────────┬───────────────┤
│  REST API    │  CLI Analysis    │  Interactive     │  Python SDK   │
│  (FastAPI)   │  (main.py)       │  (main.py)       │  (Import)     │
└──────────────┴──────────────────┴──────────────────┴───────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API Layer (api/app.py)                        │
├─────────────────────────────────────────────────────────────────────┤
│  • POST /analyze          - Analyze manual results                  │
│  • POST /analyze/image    - Upload and analyze image                │
│  • GET /health            - System health check                     │
│  • GET /                  - Service information                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
┌───────────────────────────────┐  ┌────────────────────────────────┐
│   Image Analysis Pipeline     │  │    LLM Agent Pipeline          │
│   (image_analysis/)           │  │    (llm_agent/)                │
│   ⚠️ DUMMY IMPLEMENTATION     │  │    ✅ FULL IMPLEMENTATION     │
└───────────────────────────────┘  └────────────────────────────────┘
                                               │
                                               ▼
                              ┌────────────────────────────────┐
                              │   Output: Medical Report       │
                              └────────────────────────────────┘
```

## 🔬 Image Analysis Module (UroAI) - DUMMY

```
┌─────────────────────────────────────────────────────────────────┐
│                    ImageAnalyzer (Main Class)                    │
└─────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   Glucose   │  │     pH      │  │   Nitrite   │
    │   Expert    │  │   Expert    │  │   Expert    │
    └─────────────┘  └─────────────┘  └─────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Attention Fusion│
                    │   (Multi-head)  │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Final Results  │
                    │   (JSON Output) │
                    └─────────────────┘

Example Output:
{
  "glucose": 3.1,
  "pH": 6.8,
  "nitrite": 0.2,
  "lymphocyte": 1.4,
  "UTI_probability": 0.86,
  "confidence": 0.92,
  "metadata": {...}
}
```

## 🧠 LLM Agent Module (UroGPT) - FUNCTIONAL

```
┌────────────────────────────────────────────────────────────────┐
│                      Report Generator                           │
│                    (llm_agent/generator.py)                     │
└────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│ Urinalysis   │    │  RAG Pipeline  │    │ Patient      │
│ Results      │    │  (Knowledge    │    │ Context      │
│ (JSON)       │    │   Retrieval)   │    │ (Optional)   │
└──────────────┘    └────────────────┘    └──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   LLM Engine    │
                    │  (GPT-4/Claude/ │
                    │    Gemini)      │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Medical Report  │
                    │  - Full Report  │
                    │  - Summary      │
                    │  - Interpretation│
                    │  - Recommendations│
                    └─────────────────┘
```

## 🔍 RAG Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   RAG Pipeline (rag_pipeline.py)                 │
└─────────────────────────────────────────────────────────────────┘

Step 1: Document Loading
├─ documents/sample_docs/*.txt
└─ DirectoryLoader → List[Document]

Step 2: Text Chunking
├─ RecursiveCharacterTextSplitter
├─ chunk_size: 1000
├─ chunk_overlap: 200
└─ Chunks: ~50-100 pieces

Step 3: Embedding Generation
├─ OpenAI Embeddings (API) OR
├─ HuggingFace Embeddings (Local)
└─ Vector Dimensions: 1536 (OpenAI) / 384 (HF)

Step 4: Vector Store (FAISS)
├─ Index Type: Flat L2
├─ Storage: In-memory or disk
└─ Fast similarity search

Step 5: Retrieval
├─ Query → Embedding
├─ Similarity Search (cosine/L2)
├─ Top-K Results (default: 3)
└─ Returned Documents

Step 6: Context Injection
├─ Format retrieved documents
├─ Inject into LLM prompt
└─ Generate response

Output: Retrieved Context + LLM Response
```

## 📡 API Request Flow

```
1. Client Request
   │
   ├─ POST /analyze (manual input)
   │  └─ JSON: {glucose, pH, nitrite, lymphocyte}
   │
   └─ POST /analyze/image (file upload)
      └─ Multipart: image file
   
   ↓

2. API Endpoint (api/app.py)
   │
   ├─ Validate input
   ├─ Parse parameters
   └─ Route to appropriate handler
   
   ↓

3. Processing Pipeline
   │
   ├─ Image Analysis (if image upload)
   │  └─ ImageAnalyzer.analyze() → JSON results
   │
   └─ LLM Agent
      ├─ RAGPipeline.retrieve() → Medical knowledge
      └─ ReportGenerator.generate_report() → Report
   
   ↓

4. Response
   │
   └─ JSON Response:
      {
        "status": "success",
        "urinalysis_results": {...},
        "report": "...",
        "summary": "...",
        "interpretation": {...},
        "recommendations": [...],
        "retrieved_context": [...]
      }
```

## 🔄 Data Flow Diagram

```
┌──────────────┐
│ User Input   │ (Image or Manual Data)
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│         Image Analysis (if image)           │
│  ┌────────────────────────────────────┐    │
│  │ 1. Preprocessing                   │    │
│  │ 2. Expert Models (Glucose, pH...)  │    │
│  │ 3. Attention Fusion                │    │
│  │ 4. Output Aggregation              │    │
│  └────────────────────────────────────┘    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │ JSON Results    │
         │ {glucose, pH,   │
         │  nitrite, ...}  │
         └─────────┬───────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│           LLM Agent Processing               │
│  ┌────────────────────────────────────┐     │
│  │ 1. Format Results                  │     │
│  │ 2. Query RAG for Knowledge         │     │
│  │    ├─ Search Vector Store          │     │
│  │    ├─ Retrieve Top-K Documents     │     │
│  │    └─ Format Context               │     │
│  │ 3. Build Prompt                    │     │
│  │    ├─ System Prompt                │     │
│  │    ├─ Test Results                 │     │
│  │    ├─ Retrieved Context            │     │
│  │    └─ Patient Context (optional)   │     │
│  │ 4. LLM Generation                  │     │
│  │    └─ GPT-4 / Claude / Gemini      │     │
│  │ 5. Parse & Structure Output        │     │
│  └────────────────────────────────────┘     │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │ Medical Report  │
         │ + Interpretation│
         │ + Recommendations│
         └─────────┬───────┘
                   │
                   ▼
         ┌─────────────────┐
         │  User Output    │
         │ (JSON/Text)     │
         └─────────────────┘
```

## 🗄️ Data Models

### Input: Urinalysis Results
```python
{
    "glucose": float,          # mg/dL (0-15 normal)
    "pH": float,               # 4.5-8.0 normal
    "nitrite": float,          # mg/dL (0 negative)
    "lymphocyte": float,       # cells/μL (<5 normal)
    "UTI_probability": float,  # 0-1 probability
    "confidence": float,       # 0-1 confidence
    "metadata": {
        "model_version": str,
        "processing_time_ms": int,
        "image_quality": str
    }
}
```

### Output: Medical Report
```python
{
    "report": str,             # Full NL report
    "summary": str,            # Brief summary
    "interpretation": {        # Structured interpretation
        "glucose": str,
        "pH": str,
        "nitrite": str,
        "lymphocyte": str,
        "UTI_risk": str
    },
    "recommendations": List[str],  # Action items
    "retrieved_context": List[str] # RAG sources
}
```

## 🔌 Integration Points

### 1. LLM Provider Integration
```python
# OpenAI (Default)
generator = ReportGenerator(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Anthropic Claude
generator = ReportGenerator(
    model="claude-3-opus-20240229",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Google Gemini
generator = ReportGenerator(
    model="gemini-pro",
    api_key=os.getenv("GOOGLE_API_KEY")
)
```

### 2. Embedding Model Integration
```python
# OpenAI Embeddings (Default, requires API key)
rag = RAGPipeline(embedding_model="openai")

# HuggingFace (Free, local)
rag = RAGPipeline(embedding_model="huggingface")
```

### 3. Vector Store Options
```python
# FAISS (Current, local)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Alternative: Chroma
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(chunks, embeddings)

# Alternative: Pinecone (cloud)
from langchain.vectorstores import Pinecone
vectorstore = Pinecone.from_documents(chunks, embeddings)
```

## 🔐 Security Architecture

```
┌────────────────────────────────────────┐
│        Environment Variables           │
│  (Secrets Management)                  │
├────────────────────────────────────────┤
│  • OPENAI_API_KEY                      │
│  • LLM_MODEL                           │
│  • EMBEDDING_MODEL                     │
│  • CORPUS_PATH                         │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│         API Layer (FastAPI)            │
├────────────────────────────────────────┤
│  • CORS Middleware                     │
│  • Input Validation (Pydantic)         │
│  • Error Handling                      │
│  • Rate Limiting (TODO)                │
│  • Authentication (TODO)               │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│      Business Logic Layer              │
│  (Image Analysis + LLM Agent)          │
└────────────────────────────────────────┘
```

## 📦 Deployment Architecture

### Development
```
Local Machine
├─ Python Virtual Environment
├─ FastAPI Dev Server (uvicorn)
├─ Local FAISS Vector Store
└─ API Keys in .env file
```

### Production (Recommended)
```
Cloud Platform (AWS/GCP/Azure)
├─ Container (Docker)
│  ├─ Python App
│  ├─ Dependencies
│  └─ Vector Store Cache
├─ Environment Variables (Secrets Manager)
├─ Reverse Proxy (nginx/Caddy)
│  ├─ HTTPS/TLS
│  ├─ Rate Limiting
│  └─ Load Balancing
└─ Monitoring & Logging
```

## 🔄 System States

```
┌─────────────┐
│ INITIALIZED │ (Startup)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LOADING    │ (Load models, build vector store)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   READY     │ (Accepting requests)
└──────┬──────┘
       │
       ├──────► PROCESSING ──────┐
       │                         │
       ◄─────────────────────────┘
       │
       ▼
┌─────────────┐
│   ERROR     │ (Degraded mode, fallback reports)
└─────────────┘
```

## 📈 Scalability Considerations

### Current Architecture (Single Instance)
- Handles: ~10-50 requests/second
- Memory: ~2-4 GB
- CPU: 2-4 cores sufficient

### Horizontal Scaling Options
```
Load Balancer
    │
    ├──── UroGPT Instance 1
    ├──── UroGPT Instance 2
    ├──── UroGPT Instance 3
    └──── UroGPT Instance N

Shared:
├─ Vector Store (Pinecone/Weaviate)
├─ Document Storage (S3)
└─ Cache Layer (Redis)
```

## 🎯 Technology Stack Summary

| Component | Technology | Status |
|-----------|-----------|---------|
| Image Analysis | PyTorch (placeholder) | ⚠️ Dummy |
| LLM | OpenAI GPT-4 | ✅ Functional |
| RAG Framework | LangChain | ✅ Functional |
| Vector Store | FAISS | ✅ Functional |
| Embeddings | OpenAI / HuggingFace | ✅ Functional |
| API | FastAPI | ✅ Functional |
| Server | Uvicorn | ✅ Functional |
| Data Processing | NumPy, Pandas | ✅ Functional |
| Documentation | Markdown | ✅ Complete |

---

**This architecture is designed for:**
- ✅ Modularity and maintainability
- ✅ Easy integration of real CV model
- ✅ Scalability and production deployment
- ✅ Multiple deployment options
- ✅ Clear separation of concerns
- ✅ Extensibility for future features

**Created**: November 10, 2025  
**Version**: 1.0.0  
**Status**: ✅ Production-Ready (except image module)

