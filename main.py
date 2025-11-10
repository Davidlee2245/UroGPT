#!/usr/bin/env python3
"""
UroGPT - AI-Powered Urinalysis Interpretation System
=====================================================

Main entry point for the UroGPT system.

Usage:
    # Start API server
    python main.py --mode api
    
    # Analyze single image
    python main.py --mode analyze --image path/to/image.jpg
    
    # Interactive mode
    python main.py --mode interactive

Environment Variables:
    OPENAI_API_KEY: OpenAI API key for LLM
    LLM_MODEL: Model to use (default: gpt-4)
    EMBEDDING_MODEL: Embedding model (default: openai)
    CORPUS_PATH: Path to document corpus (default: documents/sample_docs)
    PORT: API server port (default: 8000)
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from image_analysis import ImageAnalyzer
from llm_agent import RAGPipeline, ReportGenerator


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║                        UroGPT v1.0                         ║
    ║        AI-Powered Urinalysis Interpretation System         ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """Check required environment variables"""
    print("\n🔍 Checking environment...")
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for LLM functionality"
    }
    
    optional_vars = {
        "LLM_MODEL": "LLM model (default: gpt-4)",
        "EMBEDDING_MODEL": "Embedding model (default: openai)",
        "CORPUS_PATH": "Document corpus path (default: documents/sample_docs)",
        "PORT": "API server port (default: 8000)"
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            print(f"  ❌ {var}: Not set - {description}")
            missing.append(var)
        else:
            # Show partial key for security
            value = os.getenv(var)
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"  ✓ {var}: {masked}")
    
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {value}")
        else:
            print(f"  ℹ {var}: Using default - {description}")
    
    if missing:
        print(f"\n⚠️  Missing required environment variables: {', '.join(missing)}")
        print("\nPlease set them using:")
        for var in missing:
            print(f"  export {var}=your_value")
        print("\nOr create a .env file in the project root.")
        return False
    
    print("\n✓ Environment check passed!")
    return True


def run_api_mode():
    """Start FastAPI server"""
    print("\n🚀 Starting API server...")
    print("=" * 60)
    
    try:
        # Import here to avoid loading FastAPI if not needed
        from api.app import app
        import uvicorn
        
        port = int(os.getenv("PORT", 8000))
        
        print(f"\n📡 Server will be available at:")
        print(f"   - Local:   http://localhost:{port}")
        print(f"   - Docs:    http://localhost:{port}/docs")
        print(f"   - ReDoc:   http://localhost:{port}/redoc")
        print("\nPress Ctrl+C to stop the server\n")
        
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting API server: {e}")
        sys.exit(1)


def run_analyze_mode(image_path: str):
    """Analyze single image"""
    print(f"\n🔬 Analyzing image: {image_path}")
    print("=" * 60)
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Initialize components
    print("\n1. Initializing Image Analyzer...")
    image_analyzer = ImageAnalyzer()
    
    print("\n2. Initializing RAG Pipeline...")
    corpus_path = os.getenv("CORPUS_PATH", "documents/sample_docs")
    rag_pipeline = RAGPipeline(corpus_path=corpus_path)
    rag_pipeline.build_vector_store()
    
    print("\n3. Initializing Report Generator...")
    report_generator = ReportGenerator(
        model=os.getenv("LLM_MODEL", "gpt-4"),
        rag_pipeline=rag_pipeline
    )
    
    # Analyze image
    print("\n4. Analyzing urinalysis strip...")
    results = image_analyzer.analyze(image_path)
    
    print("\n" + "=" * 60)
    print("URINALYSIS RESULTS")
    print("=" * 60)
    print(f"Glucose:        {results['glucose']:.1f} mg/dL")
    print(f"pH:             {results['pH']:.1f}")
    print(f"Nitrite:        {results['nitrite']:.1f} mg/dL")
    print(f"Lymphocytes:    {results['lymphocyte']:.1f} cells/μL")
    print(f"\nUTI Probability: {results['UTI_probability']:.1%}")
    print(f"Confidence:      {results['confidence']:.1%}")
    
    # Generate report
    print("\n5. Generating medical report...")
    report_data = report_generator.generate_report(results)
    
    print("\n" + "=" * 60)
    print("MEDICAL REPORT")
    print("=" * 60)
    print(report_data["report"])
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    for i, rec in enumerate(report_data["recommendations"], 1):
        print(f"{i}. {rec}")
    
    print("\n✓ Analysis complete!")


def run_interactive_mode():
    """Interactive mode for manual input"""
    print("\n🎯 Interactive Mode")
    print("=" * 60)
    print("Enter urinalysis test results manually.\n")
    
    # Initialize components
    print("Initializing system...")
    corpus_path = os.getenv("CORPUS_PATH", "documents/sample_docs")
    rag_pipeline = RAGPipeline(corpus_path=corpus_path)
    rag_pipeline.build_vector_store()
    
    report_generator = ReportGenerator(
        model=os.getenv("LLM_MODEL", "gpt-4"),
        rag_pipeline=rag_pipeline
    )
    
    print("✓ System ready!\n")
    
    # Get input
    try:
        print("Enter test results (press Enter for default values):\n")
        
        glucose = input("Glucose (mg/dL) [default: 3.1]: ").strip()
        glucose = float(glucose) if glucose else 3.1
        
        ph = input("pH [default: 6.8]: ").strip()
        ph = float(ph) if ph else 6.8
        
        nitrite = input("Nitrite (mg/dL) [default: 0.2]: ").strip()
        nitrite = float(nitrite) if nitrite else 0.2
        
        lymphocyte = input("Lymphocytes (cells/μL) [default: 1.4]: ").strip()
        lymphocyte = float(lymphocyte) if lymphocyte else 1.4
        
        # Calculate UTI probability
        uti_score = 0
        if nitrite > 0:
            uti_score += 0.4
        if lymphocyte > 5:
            uti_score += 0.3
        if ph > 7.5:
            uti_score += 0.2
        uti_prob = min(uti_score, 1.0)
        
        results = {
            "glucose": glucose,
            "pH": ph,
            "nitrite": nitrite,
            "lymphocyte": lymphocyte,
            "UTI_probability": uti_prob,
            "confidence": 0.85
        }
        
        print("\n" + "=" * 60)
        print("ANALYZING RESULTS...")
        print("=" * 60)
        
        # Generate report
        report_data = report_generator.generate_report(results)
        
        print("\n" + "=" * 60)
        print("MEDICAL REPORT")
        print("=" * 60)
        print(report_data["report"])
        
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        for i, rec in enumerate(report_data["recommendations"], 1):
            print(f"{i}. {rec}")
        
        print("\n✓ Analysis complete!")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="UroGPT - AI-Powered Urinalysis Interpretation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start API server
  python main.py --mode api
  
  # Analyze image
  python main.py --mode analyze --image test_image.jpg
  
  # Interactive mode
  python main.py --mode interactive

Environment Variables:
  OPENAI_API_KEY     OpenAI API key (required)
  LLM_MODEL          Model to use (default: gpt-4)
  EMBEDDING_MODEL    Embedding model (default: openai)
  CORPUS_PATH        Document corpus path
  PORT               API server port (default: 8000)
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["api", "analyze", "interactive"],
        default="api",
        help="Operation mode (default: api)"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to urinalysis strip image (required for analyze mode)"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run appropriate mode
    if args.mode == "api":
        run_api_mode()
    
    elif args.mode == "analyze":
        if not args.image:
            print("❌ Error: --image required for analyze mode")
            print("Usage: python main.py --mode analyze --image path/to/image.jpg")
            sys.exit(1)
        run_analyze_mode(args.image)
    
    elif args.mode == "interactive":
        run_interactive_mode()


if __name__ == "__main__":
    main()

