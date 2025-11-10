#!/usr/bin/env python3
"""
UroGPT Example Usage
====================

This script demonstrates various ways to use the UroGPT system.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set API key if not already set
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Warning: OPENAI_API_KEY not set. Some features will not work.")
    print("Set it with: export OPENAI_API_KEY=your-key\n")


def example_1_dummy_image_analysis():
    """Example 1: Using the dummy image analyzer"""
    print("=" * 60)
    print("Example 1: Dummy Image Analysis")
    print("=" * 60)
    
    from image_analysis import ImageAnalyzer
    
    # Initialize analyzer (dummy version)
    analyzer = ImageAnalyzer()
    
    # Analyze a dummy image
    results = analyzer.analyze("fake_image.jpg")
    
    print("\nAnalysis Results:")
    print(f"  Glucose:        {results['glucose']:.1f} mg/dL")
    print(f"  pH:             {results['pH']:.1f}")
    print(f"  Nitrite:        {results['nitrite']:.1f} mg/dL")
    print(f"  Lymphocytes:    {results['lymphocyte']:.1f} cells/μL")
    print(f"  UTI Probability: {results['UTI_probability']:.1%}")
    print(f"  Confidence:     {results['confidence']:.1%}")
    
    return results


def example_2_manual_analysis():
    """Example 2: Manual analysis with custom values"""
    print("\n" + "=" * 60)
    print("Example 2: Manual Analysis with Custom Values")
    print("=" * 60)
    
    # Define custom test results
    custom_results = {
        "glucose": 5.2,      # Slightly elevated
        "pH": 7.5,           # Alkaline
        "nitrite": 0.5,      # Positive
        "lymphocyte": 8.3,   # Elevated
        "UTI_probability": 0.92,
        "confidence": 0.88
    }
    
    print("\nCustom Test Results:")
    for key, value in custom_results.items():
        print(f"  {key}: {value}")
    
    return custom_results


def example_3_rag_retrieval():
    """Example 3: Using RAG for medical knowledge retrieval"""
    print("\n" + "=" * 60)
    print("Example 3: RAG Medical Knowledge Retrieval")
    print("=" * 60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Skipped: Requires OPENAI_API_KEY")
        return
    
    try:
        from llm_agent import RAGPipeline
        
        # Initialize RAG
        rag = RAGPipeline(corpus_path="documents/sample_docs")
        rag.build_vector_store()
        
        # Query for relevant information
        query = "What does positive nitrite indicate in urinalysis?"
        docs = rag.retrieve(query, top_k=2)
        
        print(f"\nQuery: {query}")
        print(f"\nRetrieved {len(docs)} relevant documents:")
        
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Document {i} ---")
            print(doc.page_content[:300] + "...")
        
    except Exception as e:
        print(f"\n⚠️  Error: {e}")


def example_4_full_report_generation():
    """Example 4: Complete workflow with report generation"""
    print("\n" + "=" * 60)
    print("Example 4: Full Report Generation Workflow")
    print("=" * 60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Skipped: Requires OPENAI_API_KEY")
        return
    
    try:
        from image_analysis import ImageAnalyzer
        from llm_agent import RAGPipeline, ReportGenerator
        
        # Step 1: Analyze image (dummy)
        print("\n1. Analyzing urinalysis strip...")
        analyzer = ImageAnalyzer()
        results = analyzer.analyze("sample_strip.jpg")
        
        print("   ✓ Analysis complete")
        
        # Step 2: Initialize RAG
        print("\n2. Building medical knowledge base...")
        rag = RAGPipeline(corpus_path="documents/sample_docs")
        rag.build_vector_store()
        
        print("   ✓ Knowledge base ready")
        
        # Step 3: Initialize report generator
        print("\n3. Initializing report generator...")
        generator = ReportGenerator(
            model="gpt-3.5-turbo",  # Use cheaper model for demo
            rag_pipeline=rag
        )
        
        print("   ✓ Generator ready")
        
        # Step 4: Generate report
        print("\n4. Generating medical report...")
        report_data = generator.generate_report(
            urinalysis_results=results,
            use_rag=True,
            patient_context="45-year-old female with symptoms of dysuria and frequency"
        )
        
        # Display report
        print("\n" + "=" * 60)
        print("GENERATED MEDICAL REPORT")
        print("=" * 60)
        print(report_data["report"])
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(report_data["summary"])
        
        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        for param, interp in report_data["interpretation"].items():
            print(f"  {param}: {interp}")
        
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        for i, rec in enumerate(report_data["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        return report_data
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_5_batch_processing():
    """Example 5: Batch processing multiple samples"""
    print("\n" + "=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)
    
    from image_analysis import ImageAnalyzer
    
    analyzer = ImageAnalyzer()
    
    # Simulate multiple images
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    
    print(f"\nProcessing {len(image_paths)} samples...")
    results = analyzer.batch_analyze(image_paths)
    
    print("\nBatch Results Summary:")
    for i, result in enumerate(results, 1):
        uti_prob = result["UTI_probability"]
        status = "⚠️ HIGH RISK" if uti_prob > 0.7 else "✓ Low Risk"
        print(f"  Sample {i}: UTI Probability = {uti_prob:.1%} - {status}")


def example_6_api_integration():
    """Example 6: Programmatic API usage"""
    print("\n" + "=" * 60)
    print("Example 6: API Integration (Code Example)")
    print("=" * 60)
    
    api_example = '''
# Using requests library
import requests

# Analyze results
url = "http://localhost:8000/analyze"
data = {
    "glucose": 3.1,
    "pH": 6.8,
    "nitrite": 0.2,
    "lymphocyte": 1.4,
    "patient_context": "Patient history here"
}

response = requests.post(url, json=data)
result = response.json()

print(result["report"])
print(result["summary"])
print(result["interpretation"])
'''
    
    print("\nExample API Integration Code:")
    print(api_example)
    
    print("\nTo use this:")
    print("1. Start API server: python main.py --mode api")
    print("2. Run the above code in another terminal")
    print("3. Or use curl/Postman to test endpoints")


def main():
    """Run all examples"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║                UroGPT - Example Usage                      ║")
    print("║           Demonstrating System Capabilities                ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("\n")
    
    # Run examples
    try:
        results1 = example_1_dummy_image_analysis()
        results2 = example_2_manual_analysis()
        example_3_rag_retrieval()
        example_4_full_report_generation()
        example_5_batch_processing()
        example_6_api_integration()
        
        print("\n" + "=" * 60)
        print("✓ All Examples Completed!")
        print("=" * 60)
        print("\nNext Steps:")
        print("  1. Set OPENAI_API_KEY to enable full functionality")
        print("  2. Try interactive mode: python main.py --mode interactive")
        print("  3. Start API server: python main.py --mode api")
        print("  4. Read SETUP.md for detailed configuration")
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

