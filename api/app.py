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


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Starting UroGPT API on port {port}...")
    print("📝 API docs available at: http://localhost:{port}/docs\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

