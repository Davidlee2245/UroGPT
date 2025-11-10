# 🚀 UroGPT Quick Start Guide

## ⚡ 3-Minute Setup

### 1️⃣ Install Dependencies (1 minute)
```bash
cd /home/david/.cursor-tutor/UroGPT
pip install -r requirements.txt
```

### 2️⃣ Configure API Key (30 seconds)
```bash
export OPENAI_API_KEY=sk-your-actual-key-here
```

### 3️⃣ Test System (30 seconds)
```bash
python test_system.py
```

You should see:
```
✓ Basic system structure: PASS
✓ Dummy image analysis: PASS
✓ Module imports: PASS
```

### 4️⃣ Start Using! (1 minute)
```bash
# Option A: Interactive mode (easiest)
python main.py --mode interactive

# Option B: Start API server
python main.py --mode api
# Then visit: http://localhost:8000/docs
```

---

## 📋 Common Commands

### Testing
```bash
# Run system tests
python test_system.py

# Run examples
python examples/example_usage.py
```

### Running
```bash
# Interactive mode (terminal UI)
python main.py --mode interactive

# API server mode
python main.py --mode api

# Single image analysis
python main.py --mode analyze --image test.jpg
```

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Analyze results
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"glucose": 3.1, "pH": 6.8, "nitrite": 0.2, "lymphocyte": 1.4}'
```

---

## 🔧 Quick Configuration

### Environment Variables
Create `.env` file:
```bash
# Required
OPENAI_API_KEY=your-key-here

# Optional (with defaults)
LLM_MODEL=gpt-4              # or gpt-3.5-turbo
EMBEDDING_MODEL=openai       # or huggingface
CORPUS_PATH=documents/sample_docs
PORT=8000
```

### Using GPT-3.5 (Cheaper/Faster)
```bash
export LLM_MODEL=gpt-3.5-turbo
```

### Using Free Embeddings
```bash
export EMBEDDING_MODEL=huggingface
```

---

## 🐍 Python Usage

```python
from image_analysis import ImageAnalyzer
from llm_agent import RAGPipeline, ReportGenerator

# Initialize
analyzer = ImageAnalyzer()
rag = RAGPipeline()
rag.build_vector_store()
generator = ReportGenerator(rag_pipeline=rag)

# Analyze (dummy)
results = analyzer.analyze("image.jpg")

# Generate report
report = generator.generate_report(results)
print(report["report"])
```

---

## 📊 Project Structure

```
UroGPT/
├── main.py              # ← Start here
├── test_system.py       # ← Run this first
├── requirements.txt     # ← Dependencies
│
├── api/                 # REST API
├── image_analysis/      # Dummy CV module
├── llm_agent/           # LLM + RAG (functional)
├── documents/           # Medical knowledge
└── examples/            # Example scripts
```

---

## ❓ Troubleshooting

### "OpenAI API key not found"
```bash
export OPENAI_API_KEY=your-key
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Port already in use"
```bash
export PORT=8080
python main.py --mode api
```

### "Slow response"
```bash
# Use GPT-3.5 instead of GPT-4
export LLM_MODEL=gpt-3.5-turbo
```

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| `README.md` | Full documentation |
| `SETUP.md` | Detailed setup guide |
| `ARCHITECTURE.md` | Technical architecture |
| `PROJECT_SUMMARY.md` | Project overview |
| `QUICKSTART.md` | This file |

---

## 🎯 Next Steps

1. ✅ **Tested the system?** Run `test_system.py`
2. ✅ **Set API key?** Export `OPENAI_API_KEY`
3. ✅ **Try interactive mode?** `python main.py --mode interactive`
4. ✅ **Explore API?** Visit `http://localhost:8000/docs`
5. 📖 **Read full docs?** Check `README.md`

---

## 💡 Tips

- **Save costs**: Use `gpt-3.5-turbo` instead of `gpt-4`
- **Free embeddings**: Use `huggingface` embedding model
- **Add knowledge**: Put `.txt` files in `documents/sample_docs/`
- **Batch process**: Use `batch_analyze()` method
- **Cache results**: Save/load vector store with RAG

---

## 🆘 Getting Help

1. Check `SETUP.md` for detailed troubleshooting
2. Read `README.md` for full documentation
3. Run examples: `python examples/example_usage.py`
4. Check logs and error messages

---

## ⚠️ Remember

**This is for research/educational use only**
- NOT FDA approved
- NOT for clinical diagnosis
- Dummy image analysis (replace with real model)
- Always consult healthcare professionals

---

**Ready to go! 🎉**

Start with: `python main.py --mode interactive`

