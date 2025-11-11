#!/usr/bin/env python3
"""
Image Analyzer Usage Examples
==============================
Demonstrates how to use the production MobileViT-based urinalysis analyzer.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_1_basic_usage():
    """Example 1: Basic image analysis"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Image Analysis")
    print("=" * 60)
    
    from image_analysis import ImageAnalyzer
    
    # Initialize analyzer (loads model automatically)
    analyzer = ImageAnalyzer()
    
    # Analyze an image
    image_path = "path/to/urinalysis_strip.jpg"
    
    # For demonstration, we'll skip actual analysis
    print(f"\nUsage:")
    print(f"  analyzer = ImageAnalyzer()")
    print(f"  results = analyzer.analyze('{image_path}')")
    print(f"\n  # Access results")
    print(f"  print(f'UTI Probability: {{results[\"UTI_probability\"]:.1%}}')")
    print(f"  print(f'Glucose: {{results[\"glucose\"]}} mg/dL')")


def example_2_batch_processing():
    """Example 2: Batch processing multiple images"""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)
    
    print("""
from image_analysis import ImageAnalyzer

# Initialize once
analyzer = ImageAnalyzer()

# Process multiple images
image_paths = [
    'patient1_strip.jpg',
    'patient2_strip.jpg',
    'patient3_strip.jpg'
]

# Batch analyze
results_list = analyzer.batch_analyze(image_paths)

# Process results
for i, result in enumerate(results_list):
    print(f"Patient {i+1}:")
    print(f"  UTI Probability: {result['UTI_probability']:.1%}")
    print(f"  Glucose: {result['glucose']} mg/dL")
    print(f"  pH: {result['pH']}")
    print()
""")


def example_3_custom_device():
    """Example 3: Specifying device (CPU/GPU)"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Device Selection")
    print("=" * 60)
    
    print("""
from image_analysis import ImageAnalyzer

# Use specific GPU
analyzer_gpu = ImageAnalyzer(device='cuda:0')

# Use CPU (slower but no GPU required)
analyzer_cpu = ImageAnalyzer(device='cpu')

# Auto-detect (default)
analyzer_auto = ImageAnalyzer()  # Uses CUDA if available
""")


def example_4_interpreting_results():
    """Example 4: Interpreting analysis results"""
    print("\n" + "=" * 60)
    print("Example 4: Interpreting Results")
    print("=" * 60)
    
    print("""
from image_analysis import ImageAnalyzer

analyzer = ImageAnalyzer()
results = analyzer.analyze('strip.jpg')

# Clinical interpretation
def interpret_results(results):
    interpretation = []
    
    # Check glucose
    if results['glucose'] > 15:
        interpretation.append(f"⚠ Elevated glucose: {results['glucose']} mg/dL (normal: <15)")
    else:
        interpretation.append(f"✓ Normal glucose: {results['glucose']} mg/dL")
    
    # Check pH
    ph = results['pH']
    if ph < 4.5 or ph > 8.0:
        interpretation.append(f"⚠ Abnormal pH: {ph} (normal: 4.5-8.0)")
    else:
        interpretation.append(f"✓ Normal pH: {ph}")
    
    # Check nitrite
    if results['nitrite'] > 0:
        interpretation.append(f"⚠ Nitrite positive: {results['nitrite']} mg/dL (suggests bacteria)")
    else:
        interpretation.append(f"✓ Nitrite negative")
    
    # Check protein
    if results['protein'] > 30:
        interpretation.append(f"⚠ Elevated protein: {results['protein']} mg/dL (normal: <30)")
    else:
        interpretation.append(f"✓ Normal protein: {results['protein']} mg/dL")
    
    # UTI assessment
    uti_prob = results['UTI_probability']
    if uti_prob > 0.7:
        interpretation.append(f"⚠ HIGH UTI probability: {uti_prob:.1%}")
    elif uti_prob > 0.4:
        interpretation.append(f"⚠ MODERATE UTI probability: {uti_prob:.1%}")
    else:
        interpretation.append(f"✓ LOW UTI probability: {uti_prob:.1%}")
    
    return interpretation

# Generate interpretation
interpretations = interpret_results(results)
for item in interpretations:
    print(item)
""")


def example_5_api_integration():
    """Example 5: Using with FastAPI"""
    print("\n" + "=" * 60)
    print("Example 5: FastAPI Integration")
    print("=" * 60)
    
    print("""
from fastapi import FastAPI, UploadFile, File
from image_analysis import ImageAnalyzer
import tempfile

app = FastAPI()

# Initialize analyzer once at startup
analyzer = None

@app.on_event("startup")
async def startup():
    global analyzer
    analyzer = ImageAnalyzer()
    print("Analyzer ready!")

@app.post("/analyze")
async def analyze_strip(file: UploadFile = File(...)):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    # Analyze
    results = analyzer.analyze(tmp_path)
    
    # Clean up
    os.unlink(tmp_path)
    
    return {
        "status": "success",
        "results": results
    }

# Run with: uvicorn your_app:app --reload
""")


def example_6_error_handling():
    """Example 6: Error handling"""
    print("\n" + "=" * 60)
    print("Example 6: Robust Error Handling")
    print("=" * 60)
    
    print("""
from image_analysis import ImageAnalyzer
from pathlib import Path

def analyze_with_error_handling(image_path):
    try:
        # Check if file exists
        if not Path(image_path).exists():
            return {
                "status": "error",
                "message": f"Image not found: {image_path}"
            }
        
        # Check file format
        valid_formats = ['.jpg', '.jpeg', '.png']
        if Path(image_path).suffix.lower() not in valid_formats:
            return {
                "status": "error",
                "message": f"Unsupported format. Use: {valid_formats}"
            }
        
        # Initialize analyzer
        analyzer = ImageAnalyzer()
        
        # Analyze image
        results = analyzer.analyze(image_path)
        
        # Check confidence
        if results['confidence'] < 0.7:
            return {
                "status": "warning",
                "message": f"Low confidence: {results['confidence']:.1%}",
                "results": results
            }
        
        return {
            "status": "success",
            "results": results
        }
        
    except FileNotFoundError as e:
        return {"status": "error", "message": f"File error: {e}"}
    except RuntimeError as e:
        return {"status": "error", "message": f"Model error: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}

# Usage
result = analyze_with_error_handling('strip.jpg')
if result['status'] == 'success':
    print("Analysis successful!")
    print(result['results'])
elif result['status'] == 'warning':
    print(f"Warning: {result['message']}")
    print(result['results'])
else:
    print(f"Error: {result['message']}")
""")


def example_7_model_info():
    """Example 7: Accessing model information"""
    print("\n" + "=" * 60)
    print("Example 7: Model Information & Metadata")
    print("=" * 60)
    
    from image_analysis import ImageAnalyzer, MAIN_CLASSES, AUX_CLASSES_GROUPS
    
    print(f"\nMain Classes ({len(MAIN_CLASSES)}):")
    print(f"  {', '.join(MAIN_CLASSES[:5])}, ...")
    
    print(f"\nAuxiliary Groups:")
    for group_name, classes in AUX_CLASSES_GROUPS.items():
        print(f"  {group_name}: {len(classes)} classes")
    
    print(f"\nModel Architecture:")
    print(f"  - 11 specialist expert backbones (MobileViT-XS)")
    print(f"  - 6 auxiliary classifiers")
    print(f"  - 1 main classifier (33 classes)")
    print(f"  - Validation accuracy: 95.43%")
    print(f"  - Model size: ~93 MB")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print(" " * 15 + "Image Analyzer Usage Examples")
    print("=" * 70)
    
    example_1_basic_usage()
    example_2_batch_processing()
    example_3_custom_device()
    example_4_interpreting_results()
    example_5_api_integration()
    example_6_error_handling()
    example_7_model_info()
    
    print("\n" + "=" * 70)
    print("For more information, see:")
    print("  - test_analyzer.py - Comprehensive test suite")
    print("  - IMAGE_ANALYSIS_INTEGRATION.md - Full documentation")
    print("  - image_analysis/analyzer.py - Source code & docstrings")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

