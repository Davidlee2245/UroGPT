#!/bin/bash
# Start UroGPT API with Complete Pipeline
# YOLO + MobileViT analyzer

echo "=========================================="
echo "Starting UroGPT API"
echo "=========================================="
echo ""
echo "Pipeline:"
echo "  1. YOLO detects pads (yolo.pt)"
echo "  2. MobileViT analyzes (analyzer.pth)"
echo "  3. Returns clinical results"
echo ""
echo "API will be available at:"
echo "  http://localhost:8000"
echo ""
echo "Endpoints:"
echo "  GET  /              - API info"
echo "  GET  /health        - Health check"
echo "  POST /analyze       - Manual input"
echo "  POST /analyze/image - Image upload"
echo ""
echo "=========================================="
echo ""

# Start the API
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

