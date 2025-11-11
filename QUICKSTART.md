# UroGPT - Quick Start Guide

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies (2 min)

```bash
# Install Python packages
pip install -r requirements.txt

# Install Node.js 18
conda install -c conda-forge nodejs=18 -y
hash -r
```

### Step 2: Configure (1 min)

```bash
# Create environment file
cp env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use any editor
```

### Step 3: Start Backend (1 min)

```bash
./start_complete_api.sh
# Keep this terminal open
```

### Step 4: Start Web UI (1 min)

```bash
# Open new terminal
cd urogpt-ui
npm install  # first time only
npm run dev
```

### Step 5: Use It! 🎉

Open browser: **http://localhost:3000**

1. Click **"Image Analysis"**
2. Upload urinalysis strip image
3. Click **"Analyze Image"**
4. See results!

---

## 🧪 Test Without Web UI

Open in browser: `file:///home/david/.cursor-tutor/UroGPT/test_upload.html`

---

## 📊 What's Running

| Service | Port | Purpose |
|---------|------|---------|
| Backend API | 8000 | YOLO + MobileViT + GPT-4 |
| Web UI | 3000 | React interface |

---

## 🔧 Troubleshooting

**Node.js too old?**
```bash
conda install -c conda-forge nodejs=18 -y
hash -r
```

**Port already in use?**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

**Missing API key?**
```bash
# Add to .env file
OPENAI_API_KEY=your_key_here
```

---

## 📚 Full Documentation

See [README.md](README.md) for complete documentation.

---

**Pipeline**: YOLO (pad detection) + MobileViT (classification) + GPT-4 (reports)  
**Accuracy**: 95.43% validation  
**Status**: ✅ Production Ready
