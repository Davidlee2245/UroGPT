# UroGPT Setup Guide

This guide will help you get UroGPT up and running in minutes.

## 📋 Prerequisites

- **Python 3.8+** (check with `python --version`)
- **pip** package manager
- **OpenAI API key** (get one at https://platform.openai.com)
- **4GB+ RAM** recommended

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `torch` - PyTorch for image analysis placeholder
- `langchain` - RAG framework
- `openai` - OpenAI API client
- `faiss-cpu` - Vector database
- `fastapi` - Web API framework
- And other dependencies...

**Installation time:** ~2-5 minutes depending on your internet speed

### Step 2: Configure Environment

Create a `.env` file in the project root:

```bash
cp env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important:** Never commit your `.env` file to version control!

### Step 3: Test the System

Run the test script to verify everything is working:

```bash
python test_system.py
```

You should see:
```
✓ Basic system structure: PASS
✓ Dummy image analysis: PASS
✓ Module imports: PASS
```

### Step 4: Start Using UroGPT

**Option A: Interactive Mode** (easiest)
```bash
python main.py --mode interactive
```

**Option B: API Server** (for integration)
```bash
python main.py --mode api
```

Then visit: http://localhost:8000/docs

---

## 🔧 Detailed Configuration

### Environment Variables

All configuration is done via environment variables (in `.env` file):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | - | Your OpenAI API key |
| `LLM_MODEL` | No | `gpt-4` | Model to use (gpt-4, gpt-3.5-turbo) |
| `EMBEDDING_MODEL` | No | `openai` | Embedding model (openai, huggingface) |
| `CORPUS_PATH` | No | `documents/sample_docs` | Path to medical documents |
| `PORT` | No | `8000` | API server port |

### Using Different LLM Models

**GPT-3.5 (faster, cheaper):**
```env
LLM_MODEL=gpt-3.5-turbo
```

**GPT-4 (better quality):**
```env
LLM_MODEL=gpt-4
```

**Claude (Anthropic):**
```bash
pip install anthropic
```
```env
ANTHROPIC_API_KEY=your-key
LLM_MODEL=claude-3-opus-20240229
```

**Gemini (Google):**
```bash
pip install google-generativeai
```
```env
GOOGLE_API_KEY=your-key
LLM_MODEL=gemini-pro
```

### Using Free Embeddings (No API Key)

If you want to avoid API costs for embeddings:

```env
EMBEDDING_MODEL=huggingface
```

This uses the free `sentence-transformers/all-MiniLM-L6-v2` model locally.

---

## 🎯 Usage Examples

### Example 1: Interactive Analysis

```bash
python main.py --mode interactive
```

Input your values:
```
Glucose (mg/dL): 3.1
pH: 6.8
Nitrite (mg/dL): 0.2
Lymphocytes (cells/μL): 1.4
```

Get instant medical report!

### Example 2: API Integration

Start server:
```bash
python main.py --mode api
```

Test with curl:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "glucose": 3.1,
    "pH": 6.8,
    "nitrite": 0.2,
    "lymphocyte": 1.4
  }'
```

### Example 3: Python Integration

```python
from image_analysis import ImageAnalyzer
from llm_agent import RAGPipeline, ReportGenerator

# Initialize
analyzer = ImageAnalyzer()
rag = RAGPipeline()
rag.build_vector_store()
generator = ReportGenerator(rag_pipeline=rag)

# Analyze
results = {
    "glucose": 3.1,
    "pH": 6.8,
    "nitrite": 0.2,
    "lymphocyte": 1.4,
    "UTI_probability": 0.86
}

report = generator.generate_report(results)
print(report["report"])
```

---

## 📚 Adding Medical Knowledge

The RAG system uses documents in `documents/sample_docs/` for evidence-based reports.

### Add Your Own Documents

1. Create `.txt` files with medical information
2. Place in `documents/sample_docs/`
3. Restart the system

Example structure:
```
documents/sample_docs/
├── urinalysis_basics.txt
├── uti_management.txt
├── kidney_function.txt        # Your file
└── nephrology_guidelines.txt  # Your file
```

**Document Format:**
- Plain text (.txt)
- Well-structured with headings
- Include relevant medical information
- No special formatting required

---

## 🐛 Troubleshooting

### Error: "OpenAI API key not found"

**Solution:** Set the environment variable:
```bash
export OPENAI_API_KEY=your-key
```

Or add to `.env` file.

### Error: "No module named 'langchain'"

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Error: "Port 8000 already in use"

**Solution:** Use a different port:
```bash
export PORT=8080
python main.py --mode api
```

### Slow Embedding Generation

**Solution:** Switch to local embeddings:
```env
EMBEDDING_MODEL=huggingface
```

First run will download the model (~90MB), then it's cached locally.

### API Rate Limits

If you hit OpenAI rate limits:

**Option 1:** Use GPT-3.5 (cheaper):
```env
LLM_MODEL=gpt-3.5-turbo
```

**Option 2:** Add retry logic (already built-in)

**Option 3:** Switch to local models:
```bash
# Use Ollama for local LLMs
ollama pull llama3
```

---

## 🔒 Security Best Practices

1. **Never commit API keys** - They're in `.gitignore`
2. **Rotate keys periodically** - Generate new ones regularly
3. **Use environment variables** - Never hardcode secrets
4. **Enable CORS carefully** - Configure for your domain only
5. **HTTPS in production** - Use reverse proxy (nginx/Caddy)

---

## 🚢 Production Deployment

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PORT=8000
CMD ["python", "main.py", "--mode", "api"]
```

Build and run:
```bash
docker build -t urogpt .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY urogpt
```

### Cloud Deployment

**Railway:**
```bash
railway init
railway up
```

**Heroku:**
```bash
heroku create urogpt-app
git push heroku main
```

**AWS/GCP/Azure:**
Use container services (ECS, Cloud Run, Container Apps)

---

## 📊 Performance Optimization

### Reduce Latency

1. **Cache vector store:**
```python
rag_pipeline.save_vector_store("vector_store_cache")
# Load instead of rebuilding
rag_pipeline.load_vector_store("vector_store_cache")
```

2. **Use GPT-3.5 for faster responses:**
```env
LLM_MODEL=gpt-3.5-turbo
```

3. **Reduce chunk size:**
```python
rag_pipeline = RAGPipeline(chunk_size=500, top_k=2)
```

### Reduce Costs

- Use GPT-3.5 instead of GPT-4 (10x cheaper)
- Use HuggingFace embeddings (free)
- Cache responses for common queries
- Batch multiple analyses

**Typical Costs per Analysis:**
- GPT-4 + OpenAI embeddings: ~$0.05-0.10
- GPT-3.5 + OpenAI embeddings: ~$0.01-0.02
- GPT-3.5 + HuggingFace embeddings: ~$0.005-0.01

---

## 🎓 Next Steps

1. ✅ **Test the system** - Run `test_system.py`
2. ✅ **Try interactive mode** - See it in action
3. ✅ **Explore the API** - Visit `/docs` endpoint
4. 📖 **Read the README** - Full feature documentation
5. 🔬 **Integrate real model** - Replace dummy image analyzer
6. 🚀 **Deploy** - Put it in production

---

## 💬 Getting Help

- **Issues:** Check GitHub Issues
- **Docs:** Read README.md
- **Examples:** See `test_system.py` and `main.py`
- **Community:** [Your community link]

---

## ✅ Setup Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with API key
- [ ] Test script passes (`python test_system.py`)
- [ ] Interactive mode works
- [ ] API server starts successfully

**All checked?** You're ready to use UroGPT! 🎉

---

**Questions?** Open an issue on GitHub or check the FAQ in README.md

