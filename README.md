# UroGPT - AI-Powered Urinalysis Analysis System

![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

**Complete urinalysis strip analyzer using YOLO + MobileViT + GPT-4**

---

## 🎯 Overview

UroGPT is an AI-powered medical system that analyzes urinalysis strip images and generates comprehensive medical reports.

### Two-Stage Pipeline

```
Input Image
    ↓
[Stage 1] YOLO (yolo.pt)
Detects 11 sensor pads
    ↓
[Stage 2] MobileViT (analyzer.pth)
Analyzes each pad (95.43% accuracy)
    ↓
[Stage 3] GPT-4
Generates medical report
    ↓
Results + Clinical Interpretation
```

### Key Features

- 🔬 **YOLO Pad Detection**: Automatic detection of 11 sensor pads
- 🧠 **MobileViT Classification**: 95.43% validation accuracy
- 💬 **AI Medical Reports**: Natural language reports using GPT-4
- 📚 **RAG Knowledge Base**: Evidence-based medical interpretations
- 🎨 **Modern Web UI**: React-based interface
- 🚨 **UTI Risk Assessment**: Probabilistic infection detection
- 🌐 **REST API**: Easy integration

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Upgrade Node.js (for web UI)
conda install -c conda-forge nodejs=18 -y
```

### 2. Set Environment Variables

```bash
# Create .env file
cp env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_key_here
```

### 3. Start Backend API

```bash
./start_complete_api.sh
# Runs on http://localhost:8000
```

### 4. Start Web UI

```bash
cd urogpt-ui
npm install
npm run dev
# Runs on http://localhost:3000
```

### 5. Open in Browser

Go to **http://localhost:3000** and upload urinalysis strip images!

---

## 📊 Model Architecture

### Stage 1: YOLO Pad Detection

**Model**: `yolo.pt` (22 MB)  
**Purpose**: Detect 11 sensor pad positions on the strip

- **Class 0**: `urine_strip` (dipstick)
- **Class 1**: `test_pad` (11 individual pads)

### Stage 2: MobileViT Multi-Task Learning

**Model**: `analyzer.pth` (285 MB)  
**Accuracy**: 95.43% validation  
**Architecture**:
- 11 specialist expert backbones (MobileViT-XS per pad)
- 6 auxiliary classifiers:
  - **aux_0**: Hemoglobin (7 classes)
  - **aux_1**: Bilirubin (4 classes)
  - **aux_4**: Protein (6 classes)
  - **aux_5**: Nitrite (3 classes)
  - **aux_6**: Glucose (10 classes)
  - **aux_7**: pH (7 classes)
- 1 main classifier with attention fusion (33 classes)
- **Total**: 24.5M parameters

### Stage 3: GPT-4 Report Generation

**Purpose**: Generate natural language medical reports with RAG

---

## 💻 Usage

### Python API

```python
from image_analysis import ImageAnalyzer

# Initialize
analyzer = ImageAnalyzer()

# Analyze image
results = analyzer.analyze('strip_image.jpg')

# Access results
print(f"UTI Probability: {results['UTI_probability']:.1%}")
print(f"Glucose: {results['glucose']} mg/dL")
print(f"pH: {results['pH']}")
print(f"Confidence: {results['confidence']:.1%}")
```

### REST API

```bash
# Health check
curl http://localhost:8000/health

# Upload image
curl -X POST "http://localhost:8000/analyze/image" \
  -F "file=@strip.jpg"

# Manual input
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"glucose": 100, "pH": 7.0, "nitrite": 0.5}'
```

### Web Interface

1. Open http://localhost:3000
2. Go to **Image Analysis** page
3. Upload urinalysis strip image
4. Click **"Analyze Image"**
5. View results with UTI risk assessment

---

## 📁 Project Structure

```
UroGPT/
├── api/                          # FastAPI backend
│   └── app.py                    # REST API endpoints
├── image_analysis/               # YOLO + MobileViT pipeline
│   ├── analyzer.py               # Complete pipeline (700+ lines)
│   ├── yolo.pt                   # YOLO weights (22 MB)
│   ├── analyzer.pth              # MobileViT weights (285 MB)
│   └── *.ipynb                   # Training notebooks
├── llm_agent/                    # GPT-4 integration
│   ├── rag_pipeline.py           # RAG with medical documents
│   └── generator.py              # Report generation
├── urogpt-ui/                    # React web interface
│   └── src/
│       ├── pages/                # UI pages
│       └── services/             # API calls
├── documents/                    # Medical knowledge base
├── test_upload.html              # Simple test page
├── start_complete_api.sh         # Start backend script
└── requirements.txt              # Python dependencies
```

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
LLM_MODEL=gpt-4                   # or gpt-3.5-turbo
EMBEDDING_MODEL=openai            # or sentence-transformers
CORPUS_PATH=documents/sample_docs
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **YOLO Detection** | ~20-30ms (GPU) |
| **MobileViT Classification** | ~150-200ms (GPU) |
| **Total Pipeline** | ~200-250ms (GPU) |
| **Validation Accuracy** | 95.43% |
| **Model Size** | ~307 MB total |
| **Memory (GPU)** | ~2.5GB VRAM |
| **Training Images** | 4,564 strips (27,384 augmented) |

---

## 🧪 Testing

### Test Complete Pipeline

```bash
python test_analyzer.py
```

### Simple HTML Test Page

Open `test_upload.html` in browser (no Node.js needed)

### Run API Tests

```bash
# With backend running:
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Swagger UI
```

---

## 🎓 Training

### YOLO Training

See `image_analysis/yolo.ipynb`
- 1,343 training images
- Detects dipstick + 11 pads
- Output: `yolo.pt`

### MobileViT Training

See `image_analysis/9_mobilevit_xs.ipynb`
- 4,564 strips (27,384 after augmentation)
- Multi-task learning with 11 specialists
- 95.43% validation accuracy
- Output: `analyzer.pth`

---

## 🔍 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze manual input |
| `/analyze/image` | POST | Analyze uploaded image |
| `/chat` | POST | Chat with medical AI |
| `/docs` | GET | Interactive API docs (Swagger) |

---

## 🛠️ Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Node.js version too old

**Solution:**
```bash
conda install -c conda-forge nodejs=18 -y
hash -r  # Refresh terminal
```

### Issue: CUDA out of memory

**Solution:**
```python
# Use CPU instead
analyzer = ImageAnalyzer(device='cpu')
```

### Issue: React UI not starting

**Solution:**
```bash
cd urogpt-ui
rm -rf node_modules package-lock.json
npm install
npm run dev
```

---

## 📚 Documentation

- **This README**: Complete guide
- `test_upload.html`: Simple working example
- `api/app.py`: API source code with docstrings
- `image_analysis/analyzer.py`: Model pipeline with comments
- Training notebooks: `*.ipynb` files with detailed explanations

---

## 🤝 Contributing

This is a research/educational project. Feel free to:
- Test with your own urinalysis images
- Improve the models
- Add new features
- Report issues

---

## ⚠️ Disclaimer

**This is a research prototype and NOT approved for medical use.**

- Requires clinical validation before medical deployment
- Results should be verified by medical professionals
- Not a substitute for laboratory analysis
- Use for educational/research purposes only

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🎉 Credits

**Models**:
- YOLOv8 (Ultralytics)
- MobileViT-XS (Apple/timm)
- GPT-4 (OpenAI)

**Frameworks**:
- PyTorch
- FastAPI
- React + Vite

---

## 📞 Support

For issues or questions:
1. Check this README
2. Review `test_upload.html` for working example
3. Check training notebooks for model details
4. Review API documentation at `/docs`

---

**Last Updated**: November 11, 2025  
**Status**: ✅ Production Ready (requires clinical validation)  
**Pipeline**: YOLO + MobileViT + GPT-4  
**Accuracy**: 95.43% validation
