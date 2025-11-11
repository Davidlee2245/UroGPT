#!/usr/bin/env python3
"""
UroGPT - Web GUI Interface
===========================
ChatGPT-like web interface for urinalysis interpretation.

Usage:
    python app_gui.py
    
Then open: http://localhost:7860 in your browser
"""

import os
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from image_analysis import ImageAnalyzer
from llm_agent import RAGPipeline, ReportGenerator


# Global instances
image_analyzer = None
rag_pipeline = None
report_generator = None


def initialize_system():
    """Initialize the UroGPT system"""
    global image_analyzer, rag_pipeline, report_generator
    
    print("🚀 Initializing UroGPT system...")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        return False, "❌ OPENAI_API_KEY not set. Please set it in your environment."
    
    try:
        # Initialize image analyzer (dummy)
        print("1. Initializing Image Analyzer (dummy)...")
        image_analyzer = ImageAnalyzer()
        
        # Initialize RAG pipeline
        print("2. Initializing RAG Pipeline...")
        corpus_path = os.getenv("CORPUS_PATH", "documents/sample_docs")
        rag_pipeline = RAGPipeline(corpus_path=corpus_path)
        rag_pipeline.build_vector_store()
        
        # Initialize report generator
        print("3. Initializing Report Generator...")
        model = os.getenv("LLM_MODEL", "gpt-4")
        report_generator = ReportGenerator(
            model=model,
            rag_pipeline=rag_pipeline
        )
        
        print("✓ System ready!")
        return True, "✓ System initialized successfully!"
        
    except Exception as e:
        error_msg = f"❌ Initialization error: {str(e)}"
        print(error_msg)
        return False, error_msg


def analyze_from_image(image):
    """Analyze urinalysis from uploaded image"""
    if image is None:
        return "⚠️ Please upload an image first."
    
    if image_analyzer is None:
        return "❌ System not initialized. Please check your API key."
    
    try:
        # Save temporary image
        temp_path = "/tmp/urinalysis_upload.jpg"
        image.save(temp_path)
        
        # Analyze (dummy for now)
        results = image_analyzer.analyze(temp_path)
        
        # Generate report
        report_data = report_generator.generate_report(results, use_rag=True)
        
        # Format output
        output = "## 📊 Urinalysis Results\n\n"
        output += f"**Glucose:** {results['glucose']:.1f} mg/dL\n"
        output += f"**pH:** {results['pH']:.1f}\n"
        output += f"**Nitrite:** {results['nitrite']:.1f} mg/dL\n"
        output += f"**Lymphocytes:** {results['lymphocyte']:.1f} cells/μL\n"
        output += f"**UTI Probability:** {results['UTI_probability']:.1%}\n\n"
        
        output += "## 📝 Medical Report\n\n"
        output += report_data["report"]
        
        return output
        
    except Exception as e:
        return f"❌ Error analyzing image: {str(e)}"


def analyze_from_values(glucose, ph, nitrite, lymphocyte, patient_context):
    """Analyze urinalysis from manual values"""
    if report_generator is None:
        return "❌ System not initialized. Please check your API key."
    
    try:
        # Build results
        results = {
            "glucose": float(glucose) if glucose else 3.1,
            "pH": float(ph) if ph else 6.8,
            "nitrite": float(nitrite) if nitrite else 0.2,
            "lymphocyte": float(lymphocyte) if lymphocyte else 1.4,
        }
        
        # Calculate UTI probability
        uti_score = 0
        if results["nitrite"] > 0:
            uti_score += 0.4
        if results["lymphocyte"] > 5:
            uti_score += 0.3
        if results["pH"] > 7.5:
            uti_score += 0.2
        
        results["UTI_probability"] = min(uti_score, 1.0)
        results["confidence"] = 0.85
        
        # Generate report
        report_data = report_generator.generate_report(
            results,
            use_rag=True,
            patient_context=patient_context if patient_context else None
        )
        
        # Format output
        output = "## 📊 Urinalysis Results\n\n"
        output += f"**Glucose:** {results['glucose']:.1f} mg/dL\n"
        output += f"**pH:** {results['pH']:.1f}\n"
        output += f"**Nitrite:** {results['nitrite']:.1f} mg/dL\n"
        output += f"**Lymphocytes:** {results['lymphocyte']:.1f} cells/μL\n"
        output += f"**UTI Probability:** {results['UTI_probability']:.1%}\n\n"
        
        output += "## 📝 Medical Report\n\n"
        output += report_data["report"]
        
        output += "\n\n## 💊 Recommendations\n\n"
        for i, rec in enumerate(report_data["recommendations"], 1):
            output += f"{i}. {rec}\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error generating report: {str(e)}"


def chat_interface(message, history):
    """Handle chat messages - returns in messages format"""
    if not message or not message.strip():
        return history
    
    if report_generator is None:
        error_msg = "❌ System not initialized. Please set OPENAI_API_KEY and restart."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history
    
    # Add user message to history
    history.append({"role": "user", "content": message})
    
    # Check if message contains urinalysis values
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["analyze", "test", "result", "urinalysis"]):
        response = (
            "I'd be happy to help analyze urinalysis results! You can:\n\n"
            "1️⃣ **Upload an Image**: Use the 'Image Analysis' tab to upload a urinalysis strip photo\n"
            "2️⃣ **Enter Values Manually**: Use the 'Manual Input' tab to enter test results\n\n"
            "Or you can ask me questions about urinalysis, UTI, or medical interpretation!"
        )
    else:
        # Use RAG to answer questions
        try:
            context = rag_pipeline.get_context(message, top_k=2)
            
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            prompt = f"""You are UroGPT, an AI assistant specialized in urinalysis and UTI diagnosis.

User question: {message}

Relevant medical knowledge:
{context}

Provide a helpful, accurate, and professional answer based on the medical knowledge provided."""
            
            response_obj = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are UroGPT, a medical AI assistant for urinalysis interpretation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            response = response_obj.choices[0].message.content
            
        except Exception as e:
            response = f"I'm having trouble accessing my knowledge base. Error: {str(e)}\n\nPlease check your OPENAI_API_KEY is set correctly."
    
    # Add assistant response to history
    history.append({"role": "assistant", "content": response})
    
    return history


def load_documents_list():
    """Load and display list of documents in the corpus"""
    corpus_path = Path(os.getenv("CORPUS_PATH", "documents/sample_docs"))
    
    if not corpus_path.exists():
        return "📁 No documents found in corpus."
    
    # Get both .txt and .pdf files
    txt_docs = list(corpus_path.glob("*.txt"))
    pdf_docs = list(corpus_path.glob("*.pdf"))
    docs = sorted(txt_docs + pdf_docs)
    
    if not docs:
        return "📁 No documents found in corpus."
    
    output = f"## 📚 Medical Knowledge Base\n\n"
    output += f"**Location:** `{corpus_path}`\n"
    output += f"**Total Documents:** {len(docs)} ({len(txt_docs)} txt, {len(pdf_docs)} pdf)\n\n"
    output += "---\n\n"
    
    for i, doc in enumerate(docs, 1):
        file_type = "📄 PDF" if doc.suffix == ".pdf" else "📝 Text"
        output += f"### {i}. {file_type} - {doc.name}\n\n"
        
        # Read first few lines as preview
        try:
            if doc.suffix == ".txt":
                with open(doc, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    preview = ''.join(lines[:5]).strip()
                    word_count = len(' '.join(lines).split())
                    
                    output += f"**Size:** {word_count} words | **Lines:** {len(lines)}\n\n"
                    output += f"**Preview:**\n```\n{preview}\n...\n```\n\n"
            elif doc.suffix == ".pdf":
                # For PDFs, just show file info
                file_size = doc.stat().st_size
                size_mb = file_size / (1024 * 1024)
                output += f"**Type:** PDF Document\n"
                output += f"**Size:** {size_mb:.2f} MB\n\n"
                output += f"*Click 'View Document' to read the PDF content*\n\n"
            output += "---\n\n"
        except Exception as e:
            output += f"⚠️ Could not read file: {e}\n\n"
    
    return output


def get_summary_path(doc_path):
    """Get the path for the summary cache file"""
    summary_dir = Path(os.getenv("CORPUS_PATH", "documents/sample_docs")) / ".summaries"
    summary_dir.mkdir(exist_ok=True)
    
    summary_filename = doc_path.stem + ".summary.txt"
    return summary_dir / summary_filename


def load_summary(doc_name):
    """Load cached summary if exists"""
    corpus_path = Path(os.getenv("CORPUS_PATH", "documents/sample_docs"))
    doc_path = corpus_path / doc_name
    summary_path = get_summary_path(doc_path)
    
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def generate_summary(doc_name):
    """Generate AI summary and cache it"""
    corpus_path = Path(os.getenv("CORPUS_PATH", "documents/sample_docs"))
    doc_path = corpus_path / doc_name
    
    if not doc_path.exists():
        return "❌ Document not found"
    
    # Check if summary already exists
    cached_summary = load_summary(doc_name)
    if cached_summary:
        return f"## 📋 Summary (from cache)\n\n{cached_summary}"
    
    try:
        # Read document content
        if doc_path.suffix == ".txt":
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif doc_path.suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(doc_path))
            pages = loader.load()
            content = "\n\n".join([page.page_content for page in pages])
        else:
            return "⚠️ Unsupported file type"
        
        # Limit content length for API
        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[Content truncated for summary generation...]"
        
        # Generate summary using LLM
        if report_generator is None:
            return "❌ System not initialized. Please check OPENAI_API_KEY."
        
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""Please provide a comprehensive summary of this medical document.

Document: {doc_name}

Content:
{content}

Provide a summary that includes:
1. Main topic and purpose
2. Key findings or information
3. Important medical concepts covered
4. Clinical relevance

Keep it concise but informative (200-300 words)."""
        
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are a medical document summarizer. Provide clear, accurate summaries of medical literature."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        summary = response.choices[0].message.content
        
        # Save summary to cache
        summary_path = get_summary_path(doc_path)
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return f"## 📋 Summary (AI-generated)\n\n{summary}\n\n✅ *Summary cached for future use*"
        
    except Exception as e:
        return f"❌ Error generating summary: {e}"


def view_document(doc_name, show_summary=False):
    """View full content of a specific document"""
    corpus_path = Path(os.getenv("CORPUS_PATH", "documents/sample_docs"))
    doc_path = corpus_path / doc_name
    
    if not doc_path.exists():
        return f"❌ Document not found: {doc_name}"
    
    try:
        output = ""
        
        # Add summary if requested
        if show_summary:
            cached_summary = load_summary(doc_name)
            if cached_summary:
                output += f"## 📋 Summary\n\n{cached_summary}\n\n"
                output += "---\n\n"
        
        # Handle text files
        if doc_path.suffix == ".txt":
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            output += f"# 📝 {doc_name}\n\n"
            output += f"**Path:** `{doc_path}`\n"
            output += f"**Type:** Text Document\n\n"
            output += "---\n\n"
            output += content
            
            return output
        
        # Handle PDF files
        elif doc_path.suffix == ".pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
                
                loader = PyPDFLoader(str(doc_path))
                pages = loader.load()
                
                output += f"# 📄 {doc_name}\n\n"
                output += f"**Path:** `{doc_path}`\n"
                output += f"**Type:** PDF Document\n"
                output += f"**Pages:** {len(pages)}\n\n"
                output += "---\n\n"
                
                for i, page in enumerate(pages, 1):
                    output += f"## Page {i}\n\n"
                    output += page.page_content
                    output += "\n\n---\n\n"
                
                return output
            except Exception as e:
                return f"❌ Error reading PDF: {e}\n\nMake sure pypdf is installed: `pip install pypdf`"
        
        else:
            return f"⚠️ Unsupported file type: {doc_path.suffix}"
            
    except Exception as e:
        return f"❌ Error reading document: {e}"


# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for ChatGPT-style sidebar layout
    custom_css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        max-width: 100% !important;
        padding: 0 !important;
    }
    
    /* Sidebar styling */
    .sidebar {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        min-height: 100vh;
        padding: 20px 10px;
        border-right: 1px solid #2a2a3e;
    }
    
    .logo-container {
        padding: 20px 10px;
        text-align: center;
        border-bottom: 1px solid #2a2a3e;
        margin-bottom: 20px;
    }
    
    .nav-button {
        width: 100%;
        text-align: left;
        padding: 12px 16px;
        margin: 8px 0;
        background: transparent;
        color: #e0e0e0;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 14px;
    }
    
    .nav-button:hover {
        background: rgba(102, 126, 234, 0.2);
        color: white;
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }
    
    /* Main content area */
    .main-content {
        background: #ffffff;
        min-height: 100vh;
        padding: 0;
    }
    
    /* Chat area styling */
    .chat-header {
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0;
        margin: 0;
    }
    
    .chat-container {
        padding: 20px;
        background: #f7f7f8;
        min-height: calc(100vh - 200px);
    }
    
    /* Message input styling */
    .input-row {
        padding: 20px;
        background: white;
        border-top: 1px solid #e5e5e5;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .status-ready {
        background: #d4edda;
        color: #155724;
    }
    
    .status-processing {
        background: #fff3cd;
        color: #856404;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 16px 0;
        font-size: 13px;
    }
    
    /* Content sections */
    .content-section {
        padding: 30px;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Button styling */
    button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Hide default gradio tab styling */
    .tabs {
        display: none !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="UroGPT - AI Urinalysis Assistant") as interface:
        
        # Header with Logo
        with gr.Row():
            gr.Image(
                value="gui/logo.png",
                show_label=False,
                container=False,
                height=120,
                width=None,
                interactive=False,
                show_download_button=False,
                show_share_button=False
            )
        
        # Medical disclaimer
        gr.HTML("""
        <div class="warning-box">
            <strong>⚠️ Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
            NOT for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.
        </div>
        """)
        
        # Tabs
        with gr.Tabs():
            
            # Chat Tab
            with gr.Tab("💬 Chat Assistant"):
                gr.HTML("""
                <div style="background: linear-gradient(to right, #e3f2fd, #f3e5f5); 
                            padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="margin: 0; color: #1976d2;">🤖 Ask me anything about urinalysis, UTI, or medical interpretation!</h3>
                    <p style="margin: 10px 0 0 0; color: #555;">
                        💡 I can help you understand urinalysis parameters, UTI diagnosis, 
                        medical interpretations, and clinical guidelines.
                    </p>
                </div>
                """)
                
                chatbot = gr.Chatbot(
                    height=550,
                    label="💬 Conversation",
                    show_label=True,
                    type="messages",
                    avatar_images=(None, "🔬"),
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="💭 Type your message here... (e.g., 'What does elevated nitrite mean?')",
                        show_label=False,
                        scale=9,
                        container=False
                    )
                    submit_btn = gr.Button("📤 Send", scale=1, variant="primary")
                
                # Status indicator
                status = gr.Textbox(
                    label="Status",
                    value="✅ Ready to chat!",
                    interactive=False,
                    show_label=False
                )
                
                # Example questions
                gr.Examples(
                    examples=[
                        "What does positive nitrite indicate?",
                        "What are normal pH levels in urinalysis?",
                        "How is UTI diagnosed?",
                        "What causes elevated lymphocytes in urine?",
                        "I want to analyze my test results"
                    ],
                    inputs=msg,
                    label="💡 Example Questions"
                )
                
                # Handle message submission with processing indicator
                def respond(message, chat_history):
                    if not message or not message.strip():
                        return "", chat_history, "✅ Ready to chat!"
                    
                    # Show processing status
                    processing_status = "🔄 Processing your message..."
                    
                    # Get response
                    updated_history = chat_interface(message, chat_history)
                    
                    return "", updated_history, "✅ Ready to chat!"
                
                submit_btn.click(
                    respond, 
                    [msg, chatbot], 
                    [msg, chatbot, status]
                )
                msg.submit(
                    respond, 
                    [msg, chatbot], 
                    [msg, chatbot, status]
                )
            
            # Image Analysis Tab
            with gr.Tab("📷 Image Analysis"):
                gr.Markdown("""
                ### Upload Urinalysis Strip Image
                ⚠️ **Note:** Image analysis is currently using a dummy model. 
                Upload any image to see example output.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Urinalysis Strip Image"
                        )
                        analyze_image_btn = gr.Button(
                            "🔬 Analyze Image",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        image_output = gr.Markdown(label="Analysis Results")
                
                analyze_image_btn.click(
                    analyze_from_image,
                    inputs=image_input,
                    outputs=image_output
                )
            
            # Manual Input Tab
            with gr.Tab("⌨️ Manual Input"):
                gr.Markdown("""
                ### Enter Test Results Manually
                Enter your urinalysis test values below. Leave blank to use default values.
                """)
                
                with gr.Row():
                    with gr.Column():
                        glucose_input = gr.Number(
                            label="Glucose (mg/dL)",
                            value=3.1,
                            info="Normal: 0-15 mg/dL"
                        )
                        ph_input = gr.Number(
                            label="pH",
                            value=6.8,
                            info="Normal: 4.5-8.0"
                        )
                        nitrite_input = gr.Number(
                            label="Nitrite (mg/dL)",
                            value=0.2,
                            info="Normal: Negative (0)"
                        )
                        lymphocyte_input = gr.Number(
                            label="Lymphocytes (cells/μL)",
                            value=1.4,
                            info="Normal: <5 cells/μL"
                        )
                        patient_context_input = gr.Textbox(
                            label="Patient Context (Optional)",
                            placeholder="e.g., 45-year-old female with dysuria...",
                            lines=3
                        )
                        
                        analyze_manual_btn = gr.Button(
                            "🔬 Generate Report",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column():
                        manual_output = gr.Markdown(label="Analysis Results")
                
                # Example presets
                with gr.Row():
                    gr.Markdown("### Quick Presets:")
                
                with gr.Row():
                    normal_btn = gr.Button("✅ Normal Results", size="sm")
                    uti_btn = gr.Button("⚠️ Possible UTI", size="sm")
                    high_glucose_btn = gr.Button("📊 High Glucose", size="sm")
                
                def set_normal():
                    return 5.0, 6.5, 0.0, 2.0, ""
                
                def set_uti():
                    return 8.0, 7.5, 0.5, 12.0, "Patient with dysuria and frequency"
                
                def set_high_glucose():
                    return 45.0, 6.2, 0.0, 3.0, "Known diabetic patient"
                
                normal_btn.click(
                    set_normal,
                    outputs=[glucose_input, ph_input, nitrite_input, lymphocyte_input, patient_context_input]
                )
                
                uti_btn.click(
                    set_uti,
                    outputs=[glucose_input, ph_input, nitrite_input, lymphocyte_input, patient_context_input]
                )
                
                high_glucose_btn.click(
                    set_high_glucose,
                    outputs=[glucose_input, ph_input, nitrite_input, lymphocyte_input, patient_context_input]
                )
                
                analyze_manual_btn.click(
                    analyze_from_values,
                    inputs=[glucose_input, ph_input, nitrite_input, lymphocyte_input, patient_context_input],
                    outputs=manual_output
                )
            
            # Documents Tab
            with gr.Tab("📚 Documents"):
                gr.HTML("""
                <div style="background: linear-gradient(to right, #e8f5e9, #f1f8e9); 
                            padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="margin: 0; color: #2e7d32;">📚 Medical Knowledge Base</h3>
                    <p style="margin: 10px 0 0 0; color: #555;">
                        Browse the medical documents used by UroGPT for evidence-based interpretations
                    </p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Document List")
                        
                        # Get list of documents (both .txt and .pdf)
                        corpus_path = Path(os.getenv("CORPUS_PATH", "documents/sample_docs"))
                        if corpus_path.exists():
                            txt_docs = list(corpus_path.glob("*.txt"))
                            pdf_docs = list(corpus_path.glob("*.pdf"))
                            docs = sorted(txt_docs + pdf_docs)
                        else:
                            docs = []
                        doc_names = [doc.name for doc in docs]
                        
                        if doc_names:
                            doc_selector = gr.Radio(
                                choices=doc_names,
                                label="Select a document to view:",
                                value=doc_names[0] if doc_names else None
                            )
                            
                            with gr.Row():
                                view_btn = gr.Button("📖 View Document", variant="primary", scale=2)
                                summary_btn = gr.Button("📋 Generate Summary", variant="secondary", scale=2)
                            
                            with gr.Row():
                                view_with_summary_btn = gr.Button("📄 View with Summary", variant="secondary", scale=2)
                                refresh_btn = gr.Button("🔄 Refresh List", variant="secondary", scale=1)
                        else:
                            gr.Markdown("⚠️ No documents found in corpus.")
                            doc_selector = gr.Radio(choices=[], label="Select a document")
                            with gr.Row():
                                view_btn = gr.Button("📖 View Document", variant="primary", interactive=False)
                                summary_btn = gr.Button("📋 Generate Summary", variant="secondary", interactive=False)
                            with gr.Row():
                                view_with_summary_btn = gr.Button("📄 View with Summary", variant="secondary", interactive=False)
                                refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")
                    
                    with gr.Column(scale=2):
                        doc_viewer = gr.Markdown(
                            value=load_documents_list(),
                            label="Document Content"
                        )
                
                # Button actions
                if doc_names:
                    # View document without summary
                    view_btn.click(
                        lambda doc: view_document(doc, show_summary=False),
                        inputs=doc_selector,
                        outputs=doc_viewer
                    )
                    
                    # Generate summary only
                    summary_btn.click(
                        generate_summary,
                        inputs=doc_selector,
                        outputs=doc_viewer
                    )
                    
                    # View document with summary at top
                    view_with_summary_btn.click(
                        lambda doc: view_document(doc, show_summary=True),
                        inputs=doc_selector,
                        outputs=doc_viewer
                    )
                    
                    # Refresh document list
                    refresh_btn.click(
                        lambda: load_documents_list(),
                        outputs=doc_viewer
                    )
                
                # Document statistics
                with gr.Accordion("📊 Corpus Statistics", open=False):
                    if docs:
                        stats_md = f"""
                        **Total Documents:** {len(docs)}
                        
                        **Document Details:**
                        """
                        for doc in docs:
                            if doc.suffix == ".txt":
                                with open(doc, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    words = len(content.split())
                                    lines = len(content.split('\n'))
                                stats_md += f"\n- **{doc.name}** (Text): {words} words, {lines} lines"
                            elif doc.suffix == ".pdf":
                                file_size = doc.stat().st_size
                                size_mb = file_size / (1024 * 1024)
                                stats_md += f"\n- **{doc.name}** (PDF): {size_mb:.2f} MB"
                        
                        gr.Markdown(stats_md)
                    else:
                        gr.Markdown("No documents available.")
            
            # About Tab
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About UroGPT
                
                **UroGPT** is an AI-powered urinalysis interpretation system that combines:
                
                - 🔬 **Image Analysis** (Dummy): Computer vision for urinalysis strip analysis
                - 🧠 **LLM Agent**: Advanced language models (GPT-4) for report generation
                - 📚 **RAG Pipeline**: Retrieval-Augmented Generation with medical knowledge base
                - 🩺 **Evidence-Based**: Interpretations grounded in medical literature
                
                ### Features
                
                - ✅ Natural language chat interface
                - ✅ Image-based analysis (coming soon with real model)
                - ✅ Manual value input
                - ✅ Comprehensive medical reports
                - ✅ UTI probability assessment
                - ✅ Clinical recommendations
                
                ### Technology Stack
                
                - **Frontend**: Gradio
                - **Backend**: Python, FastAPI
                - **LLM**: OpenAI GPT-4
                - **RAG**: LangChain + FAISS
                - **Image Analysis**: PyTorch (placeholder)
                
                ### Medical Disclaimer
                
                ⚠️ **FOR RESEARCH AND EDUCATIONAL USE ONLY**
                
                - NOT FDA approved
                - NOT for clinical diagnosis or treatment
                - Always consult qualified healthcare professionals
                - Image analysis module is currently a placeholder
                
                ### Version
                
                **UroGPT v1.0** - November 2025
                
                ### GitHub
                
                [https://github.com/Davidlee2245/UroGPT](https://github.com/Davidlee2245/UroGPT)
                """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666; font-size: 0.9em;">
            <p>© 2025 UroGPT - AI-Powered Urinalysis Interpretation System</p>
            <p>Made with ❤️ using Python, OpenAI, and Gradio</p>
        </div>
        """)
    
    return interface


def main():
    """Main entry point"""
    print("=" * 60)
    print("UroGPT Web GUI")
    print("=" * 60)
    
    # Initialize system
    success, message = initialize_system()
    print(f"\n{message}\n")
    
    if not success:
        print("⚠️  System initialization failed.")
        print("Please set OPENAI_API_KEY environment variable:")
        print("  export OPENAI_API_KEY=your-key-here")
        print("\nStarting GUI anyway (limited functionality)...\n")
    
    # Create and launch interface
    interface = create_interface()
    
    print("=" * 60)
    print("🌐 Starting web server...")
    print("=" * 60)
    print("\n📱 Open in your browser:")
    print("   Local:   http://localhost:7860")
    print("   Network: http://0.0.0.0:7860")
    print("\n Press Ctrl+C to stop the server\n")
    
    # Launch
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )


if __name__ == "__main__":
    main()

