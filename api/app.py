"""
FastAPI Application
===================
REST API endpoints for urinalysis analysis and report generation.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_analysis import ImageAnalyzer
from llm_agent import RAGPipeline, ReportGenerator


# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    """Request model for manual result input"""
    glucose: Optional[float] = Field(None, description="Glucose level in mg/dL")
    pH: Optional[float] = Field(None, description="pH level (4.5-8.0)")
    nitrite: Optional[float] = Field(None, description="Nitrite level in mg/dL")
    lymphocyte: Optional[float] = Field(None, description="Lymphocyte count in cells/μL")
    patient_context: Optional[str] = Field(None, description="Additional patient context")


class AnalysisResponse(BaseModel):
    """Response model for analysis"""
    status: str
    urinalysis_results: Dict[str, Any]
    report: str
    summary: str
    interpretation: Dict[str, str]
    recommendations: list
    retrieved_context: Optional[list] = None


# Initialize FastAPI app
app = FastAPI(
    title="UroGPT API",
    description="AI-powered urinalysis interpretation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
image_analyzer = None
rag_pipeline = None
report_generator = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and pipelines on startup"""
    global image_analyzer, rag_pipeline, report_generator
    
    print("=" * 60)
    print("🚀 Starting UroGPT API...")
    print("=" * 60)
    
    # Initialize image analyzer
    print("\n1. Initializing Image Analyzer (dummy)...")
    image_analyzer = ImageAnalyzer()
    
    # Initialize RAG pipeline
    print("\n2. Initializing RAG Pipeline...")
    corpus_path = os.getenv("CORPUS_PATH", "documents/sample_docs")
    embedding_model = os.getenv("EMBEDDING_MODEL", "openai")
    
    try:
        rag_pipeline = RAGPipeline(
            corpus_path=corpus_path,
            embedding_model=embedding_model
        )
        rag_pipeline.build_vector_store()
    except Exception as e:
        print(f"⚠️  RAG initialization warning: {e}")
        print("Continuing without RAG support...")
        rag_pipeline = None
    
    # Initialize report generator
    print("\n3. Initializing Report Generator...")
    model = os.getenv("LLM_MODEL", "gpt-4")
    
    try:
        report_generator = ReportGenerator(
            model=model,
            rag_pipeline=rag_pipeline
        )
    except Exception as e:
        print(f"❌ Report generator initialization failed: {e}")
        print("Please set OPENAI_API_KEY environment variable")
        report_generator = None
    
    print("\n" + "=" * 60)
    print("✓ UroGPT API Ready!")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  - POST /analyze - Analyze with manual results")
    print("  - POST /analyze/image - Analyze from image")
    print("  - GET /health - Health check")
    print("\n")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "UroGPT API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "/analyze",
            "analyze_image": "/analyze/image",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "image_analyzer": "ready" if image_analyzer else "not initialized",
            "rag_pipeline": "ready" if rag_pipeline else "not available",
            "report_generator": "ready" if report_generator else "not initialized"
        }
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_results(request: AnalysisRequest):
    """
    Analyze urinalysis results and generate report.
    
    Accepts manual input of test results.
    """
    if not report_generator:
        raise HTTPException(
            status_code=503,
            detail="Report generator not initialized. Check OPENAI_API_KEY."
        )
    
    # Build results dictionary
    urinalysis_results = {}
    
    if request.glucose is not None:
        urinalysis_results["glucose"] = request.glucose
    if request.pH is not None:
        urinalysis_results["pH"] = request.pH
    if request.nitrite is not None:
        urinalysis_results["nitrite"] = request.nitrite
    if request.lymphocyte is not None:
        urinalysis_results["lymphocyte"] = request.lymphocyte
    
    # If no results provided, return error
    if not urinalysis_results:
        raise HTTPException(
            status_code=400,
            detail="No test results provided"
        )
    
    # Calculate dummy UTI probability based on parameters
    uti_score = 0
    if urinalysis_results.get("nitrite", 0) > 0:
        uti_score += 0.4
    if urinalysis_results.get("lymphocyte", 0) > 5:
        uti_score += 0.3
    if urinalysis_results.get("pH", 7) > 7.5:
        uti_score += 0.2
    
    urinalysis_results["UTI_probability"] = min(uti_score, 1.0)
    urinalysis_results["confidence"] = 0.85
    
    # Generate report
    try:
        report_data = report_generator.generate_report(
            urinalysis_results=urinalysis_results,
            use_rag=rag_pipeline is not None,
            patient_context=request.patient_context
        )
        
        return AnalysisResponse(
            status="success",
            urinalysis_results=urinalysis_results,
            report=report_data["report"],
            summary=report_data["summary"],
            interpretation=report_data["interpretation"],
            recommendations=report_data["recommendations"],
            retrieved_context=report_data.get("retrieved_context")
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating report: {str(e)}"
        )


@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    patient_context: Optional[str] = None
):
    """
    Analyze urinalysis strip image and generate report.
    
    Uploads image to UroAI for analysis, then generates natural language report.
    """
    if not image_analyzer:
        raise HTTPException(
            status_code=503,
            detail="Image analyzer not initialized"
        )
    
    if not report_generator:
        raise HTTPException(
            status_code=503,
            detail="Report generator not initialized. Check OPENAI_API_KEY."
        )
    
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Analyze image (dummy analysis)
        print(f"Analyzing image: {file.filename}")
        urinalysis_results = image_analyzer.analyze(temp_path)
        
        # Generate report
        report_data = report_generator.generate_report(
            urinalysis_results=urinalysis_results,
            use_rag=rag_pipeline is not None,
            patient_context=patient_context
        )
        
        return AnalysisResponse(
            status="success",
            urinalysis_results=urinalysis_results,
            report=report_data["report"],
            summary=report_data["summary"],
            interpretation=report_data["interpretation"],
            recommendations=report_data["recommendations"],
            retrieved_context=report_data.get("retrieved_context")
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}"
        )
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/chat")
async def chat(request: AnalysisRequest):
    """
    Chat endpoint for general queries without requiring test results.
    """
    if not report_generator:
        raise HTTPException(
            status_code=503,
            detail="Report generator not initialized. Check OPENAI_API_KEY."
        )
    
    if not request.patient_context:
        raise HTTPException(
            status_code=400,
            detail="No query provided"
        )
    
    try:
        # Use dummy results to trigger report generation with just the query
        dummy_results = {"query_mode": True}
        
        report_data = report_generator.generate_report(
            urinalysis_results=dummy_results,
            use_rag=rag_pipeline is not None,
            patient_context=request.patient_context
        )
        
        return {
            "status": "success",
            "response": report_data["report"]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@app.get("/documents")
async def list_documents():
    """
    List all documents in the corpus.
    """
    try:
        corpus_path = os.getenv("CORPUS_PATH", "documents/sample_docs")
        docs = []
        
        if os.path.exists(corpus_path):
            for file in Path(corpus_path).glob("*"):
                if file.is_file() and file.suffix.lower() in ['.txt', '.pdf']:
                    docs.append({
                        "filename": file.name,
                        "filepath": str(file),
                        "type": file.suffix[1:].lower(),
                        "size": file.stat().st_size
                    })
        
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.get("/documents/content")
async def get_document_content(filepath: str):
    """
    Get the content of a specific document.
    """
    try:
        file_path = Path(filepath)
        
        # Security check: ensure file is within documents directory
        corpus_path = Path(os.getenv("CORPUS_PATH", "documents/sample_docs"))
        if not str(file_path.resolve()).startswith(str(corpus_path.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Read file based on type
        if file_path.suffix.lower() == '.pdf':
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            content = "\n\n".join([doc.page_content for doc in documents])
        else:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        return {"content": content, "filename": file_path.name}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading document: {str(e)}")


def get_summary_path(doc_path: Path) -> Path:
    """Get the path where summary should be stored."""
    summary_dir = doc_path.parent / ".summaries"
    summary_dir.mkdir(exist_ok=True)
    # Use .summary.txt format to match existing files from app_gui.py
    return summary_dir / f"{doc_path.stem}.summary.txt"


@app.get("/documents/summary")
async def get_cached_summary(filepath: str):
    """
    Get cached summary for a document if it exists.
    """
    try:
        file_path = Path(filepath)
        
        # Security check
        corpus_path = Path(os.getenv("CORPUS_PATH", "documents/sample_docs"))
        if not str(file_path.resolve()).startswith(str(corpus_path.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        summary_path = get_summary_path(file_path)
        
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = f.read()
            return {"summary": summary, "cached": True}
        else:
            return {"summary": None, "cached": False}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading summary: {str(e)}")


class SummarySaveRequest(BaseModel):
    """Request model for saving summary"""
    filepath: str
    summary: str


@app.post("/documents/summary")
async def save_summary(request: SummarySaveRequest):
    """
    Save generated summary to cache.
    """
    try:
        file_path = Path(request.filepath)
        
        # Security check
        corpus_path = Path(os.getenv("CORPUS_PATH", "documents/sample_docs"))
        if not str(file_path.resolve()).startswith(str(corpus_path.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        summary_path = get_summary_path(file_path)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(request.summary)
        
        return {"status": "success", "message": "Summary cached successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving summary: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Starting UroGPT API on port {port}...")
    print("📝 API docs available at: http://localhost:{port}/docs\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

