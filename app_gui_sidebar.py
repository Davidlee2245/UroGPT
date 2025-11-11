"""
UroGPT Web GUI with ChatGPT-style Sidebar Layout
Version: 2.0 (Sidebar Design)
"""

import os
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import modules
from image_analysis.analyzer import ImageAnalyzer
from llm_agent.rag_pipeline import RAGPipeline
from llm_agent.generator import ReportGenerator

# Global variables for system components
image_analyzer = None
rag_pipeline = None
report_generator = None


def initialize_system():
    """Initialize all system components"""
    global image_analyzer, rag_pipeline, report_generator
    
    print("="*60)
    print("UroGPT Web GUI - ChatGPT Style")
    print("="*60)
    print("🚀 Initializing UroGPT system...")
    
    # Initialize Image Analyzer (dummy)
    print("1. Initializing Image Analyzer (dummy)...")
    image_analyzer = ImageAnalyzer()
    
    # Initialize RAG Pipeline
    print("2. Initializing RAG Pipeline...")
    corpus_path = os.getenv("CORPUS_PATH", "documents/sample_docs")
    rag_pipeline = RAGPipeline(
        corpus_path=corpus_path
    )
    
    # Load documents and build vector store
    docs = rag_pipeline.load_documents()
    if docs:
        rag_pipeline.build_vector_store(docs)
    
    # Initialize Report Generator
    print("3. Initializing Report Generator...")
    llm_model = os.getenv("LLM_MODEL", "gpt-4")
    report_generator = ReportGenerator(
        model=llm_model,
        rag_pipeline=rag_pipeline
    )
    print("✓ System ready!")
    print()
    print("✓ System initialized successfully!")
    print()


# Chat interface function
def chat_interface(message, history):
    """Handle chat messages"""
    if not message or not message.strip():
        return history
    
    # Add user message
    history.append({"role": "user", "content": message})
    
    # Check if this is an analysis request
    if any(word in message.lower() for word in ["analyze", "test", "results", "report", "interpret"]):
        response = "I can help analyze urinalysis results! You can:\n\n"
        response += "1. Use the **Image Analysis** section to upload a urinalysis strip image\n"
        response += "2. Use the **Manual Input** section to enter test values manually\n\n"
        response += "Would you like to provide your test results?"
    else:
        # Get context from RAG
        context = rag_pipeline.get_context(message) if rag_pipeline else ""
        
        # Generate response
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        system_prompt = "You are UroGPT, an AI assistant specialized in urinalysis and urinary tract infection (UTI) diagnosis. Provide clear, accurate medical information."
        if context:
            system_prompt += f"\n\nRelevant medical knowledge:\n{context}"
        
        response_obj = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": system_prompt},
                *history
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        response = response_obj.choices[0].message.content
    
    # Add assistant response
    history.append({"role": "assistant", "content": response})
    
    return history


# Analysis functions
def analyze_from_image(image):
    """Analyze from uploaded image"""
    if image is None:
        return "⚠️ Please upload an image first."
    
    # Dummy analysis
    results = image_analyzer.analyze(image)
    report = report_generator.generate_report(results)
    
    return f"## 🔬 Analysis Results\n\n{report}"


def analyze_from_values(glucose, ph, nitrite, lymphocyte, patient_context):
    """Analyze from manual input values"""
    results = {
        "glucose": glucose,
        "pH": ph,
        "nitrite": nitrite,
        "lymphocyte": lymphocyte,
        "UTI_probability": 0.65 if nitrite > 0.3 or lymphocyte > 10 else 0.15
    }
    
    report = report_generator.generate_report(results, patient_context)
    
    return f"## 🔬 Analysis Results\n\n{report}"


# Document functions (from original code)
def load_documents_list():
    """Load and display list of documents"""
    corpus_path = Path(os.getenv("CORPUS_PATH", "documents/sample_docs"))
    
    if not corpus_path.exists():
        return "📁 No documents found in corpus."
    
    txt_docs = list(corpus_path.glob("*.txt"))
    pdf_docs = list(corpus_path.glob("*.pdf"))
    docs = sorted(txt_docs + pdf_docs)
    
    if not docs:
        return "📁 No documents found in corpus."
    
    output = f"# 📚 Medical Knowledge Base\n\n**Total Documents:** {len(docs)}\n\n---\n\n"
    
    for i, doc in enumerate(docs, 1):
        file_type = "📄 PDF" if doc.suffix == ".pdf" else "📝 Text"
        output += f"### {i}. {file_type} - {doc.name}\n\n"
        
        try:
            if doc.suffix == ".txt":
                with open(doc, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    preview = ''.join(lines[:5]).strip()
                    word_count = len(' '.join(lines).split())
                    
                    output += f"**Size:** {word_count} words | **Lines:** {len(lines)}\n\n"
                    output += f"**Preview:**\n```\n{preview}\n...\n```\n\n"
            elif doc.suffix == ".pdf":
                file_size = doc.stat().st_size
                size_mb = file_size / (1024 * 1024)
                output += f"**Type:** PDF Document\n"
                output += f"**Size:** {size_mb:.2f} MB\n\n"
                output += f"*Click 'View Document' to read the PDF content*\n\n"
            output += "---\n\n"
        except Exception as e:
            output += f"⚠️ Could not read file: {e}\n\n"
    
    return output


def view_document(doc_name, show_summary=False):
    """View document content"""
    corpus_path = Path(os.getenv("CORPUS_PATH", "documents/sample_docs"))
    doc_path = corpus_path / doc_name
    
    if not doc_path.exists():
        return f"❌ Document not found: {doc_name}"
    
    try:
        output = ""
        
        if doc_path.suffix == ".txt":
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            output += f"# 📝 {doc_name}\n\n"
            output += f"**Path:** `{doc_path}`\n"
            output += f"**Type:** Text Document\n\n"
            output += "---\n\n"
            output += content
            
            return output
        
        elif doc_path.suffix == ".pdf":
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
        
        else:
            return f"⚠️ Unsupported file type: {doc_path.suffix}"
            
    except Exception as e:
        return f"❌ Error reading document: {e}"


# Create interface
def create_interface():
    """Create ChatGPT-style sidebar interface"""
    
    # CSS for Professional layout with logo teal theme
    custom_css = """
    /* Global styling - Sans-serif everywhere */
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    /* Sidebar - Logo teal color theme */
    #sidebar {
        background: linear-gradient(180deg, #10a37f 0%, #0d8c6f 100%) !important;
        min-height: 100vh;
        padding: 0 !important;
        border-right: 1px solid #0a7a5f;
        box-shadow: 2px 0 12px rgba(16, 163, 127, 0.2);
    }
    
    .logo-section {
        padding: 24px 16px;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        background: rgba(255,255,255,0.1);
    }
    
    .nav-section {
        padding: 12px 8px;
    }
    
    /* Main content - Clean white */
    #main-content {
        background: #ffffff !important;
        min-height: 100vh;
        padding: 0 !important;
    }
    
    /* Navigation buttons - Clean professional style */
    .nav-btn {
        margin: 4px 0 !important;
        padding: 12px 16px !important;
        text-align: left !important;
        border-radius: 8px !important;
        background: transparent !important;
        color: rgba(255,255,255,0.9) !important;
        border: none !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    
    .nav-btn:hover {
        background: rgba(255,255,255,0.15) !important;
        color: #ffffff !important;
    }
    
    /* Active navigation button - White background */
    button[variant="primary"] {
        background: #ffffff !important;
        color: #10a37f !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    button[variant="primary"]:hover {
        background: #f9fafb !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* Header sections */
    .page-header {
        background: linear-gradient(135deg, #10a37f 0%, #0e8c6f 100%) !important;
        padding: 32px 40px !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    .page-header h2 {
        margin: 0 !important;
        font-size: 24px !important;
        font-weight: 600 !important;
    }
    
    .page-header p {
        margin: 8px 0 0 0 !important;
        opacity: 0.95 !important;
        font-size: 14px !important;
    }
    
    /* Content area */
    .content-area {
        padding: 32px 40px !important;
        max-width: 1000px !important;
        margin: 0 auto !important;
        background: #ffffff !important;
    }
    
    /* Chat interface */
    .chatbot {
        border: 1px solid #e5e5e5 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
        background: #ffffff !important;
    }
    
    /* Input area */
    textarea, input[type="text"] {
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }
    
    textarea:focus, input[type="text"]:focus {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.1) !important;
        outline: none !important;
    }
    
    /* All buttons - Logo color theme */
    button {
        font-family: 'Inter', sans-serif !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    /* Primary action buttons */
    button[variant="primary"]:not(.nav-btn) {
        background: #10a37f !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        font-size: 14px !important;
    }
    
    button[variant="primary"]:not(.nav-btn):hover {
        background: #0d8c6f !important;
        box-shadow: 0 4px 12px rgba(16, 163, 127, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Examples section */
    .examples {
        background: #f9fafb !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }
    
    /* Cards and panels */
    .panel {
        background: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #10a37f;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #0e8c6f;
    }
    
    /* Remove default gradio styling */
    .svelte-1ed2p3z {
        padding: 0 !important;
    }
    
    /* Smooth transitions */
    * {
        transition: background-color 0.2s ease, border-color 0.2s ease !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="UroGPT - AI Assistant") as interface:
        
        # Track current page
        current_page = gr.State("chat")
        
        with gr.Row():
            # Sidebar (20% width)
            with gr.Column(scale=2, elem_id="sidebar"):
                # Logo
                with gr.Column(elem_classes="logo-section"):
                    gr.Image(
                        value="gui/logo.png",
                        show_label=False,
                        container=False,
                        height=80,
                        interactive=False,
                        show_download_button=False,
                        show_share_button=False
                    )
                
                # Navigation buttons
                with gr.Column(elem_classes="nav-section"):
                    chat_nav = gr.Button("Chat Assistant", elem_classes="nav-btn", variant="primary", size="lg")
                    image_nav = gr.Button("Image Analysis", elem_classes="nav-btn", size="lg")
                    manual_nav = gr.Button("Manual Input", elem_classes="nav-btn", size="lg")
                    docs_nav = gr.Button("Documents", elem_classes="nav-btn", size="lg")
                    about_nav = gr.Button("About", elem_classes="nav-btn", size="lg")
            
            # Main content area (80% width)
            with gr.Column(scale=8, elem_id="main-content"):
                
                # Chat page
                chat_section = gr.Column(visible=True)
                with chat_section:
                    gr.HTML("""
                    <div class='page-header'>
                        <h2>Chat Assistant</h2>
                        <p>Ask me anything about urinalysis, UTI, or medical interpretation!</p>
                    </div>
                    """)
                    
                    with gr.Column(elem_classes="content-area"):
                        chatbot = gr.Chatbot(
                            height=500,
                            type="messages",
                            show_copy_button=True
                        )
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Type your message...",
                                show_label=False,
                                scale=9,
                                container=False
                            )
                            submit_btn = gr.Button("Send", scale=1, variant="primary")
                        
                        gr.Examples(
                            examples=[
                                "What does positive nitrite indicate?",
                                "What are normal pH levels in urinalysis?",
                                "How is UTI diagnosed?"
                            ],
                            inputs=msg
                        )
                
                # Image Analysis page
                image_section = gr.Column(visible=False)
                with image_section:
                    gr.HTML("""
                    <div class='page-header'>
                        <h2>Image Analysis</h2>
                        <p>Upload urinalysis strip image for AI analysis</p>
                    </div>
                    """)
                    
                    with gr.Column(elem_classes="content-area"):
                        with gr.Row():
                            with gr.Column():
                                image_input = gr.Image(type="pil", label="Upload Image")
                                analyze_image_btn = gr.Button("Analyze", variant="primary", size="lg")
                            
                            with gr.Column():
                                image_output = gr.Markdown()
                
                # Manual Input page
                manual_section = gr.Column(visible=False)
                with manual_section:
                    gr.HTML("""
                    <div class='page-header'>
                        <h2>Manual Input</h2>
                        <p>Enter your test results manually</p>
                    </div>
                    """)
                    
                    with gr.Column(elem_classes="content-area"):
                        with gr.Row():
                            with gr.Column():
                                glucose_input = gr.Number(label="Glucose (mg/dL)", value=3.1)
                                ph_input = gr.Number(label="pH", value=6.8)
                                nitrite_input = gr.Number(label="Nitrite (mg/dL)", value=0.2)
                                lymphocyte_input = gr.Number(label="Lymphocytes (cells/μL)", value=1.4)
                                patient_context_input = gr.Textbox(
                                    label="Patient Context (Optional)",
                                    lines=3
                                )
                                analyze_manual_btn = gr.Button("Generate Report", variant="primary", size="lg")
                            
                            with gr.Column():
                                manual_output = gr.Markdown()
                
                # Documents page
                docs_section = gr.Column(visible=False)
                with docs_section:
                    gr.HTML("""
                    <div class='page-header'>
                        <h2>Medical Knowledge Base</h2>
                        <p>Browse medical documents used for evidence-based interpretation</p>
                    </div>
                    """)
                    
                    with gr.Column(elem_classes="content-area"):
                        with gr.Row():
                            with gr.Column(scale=1):
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
                                        label="Select a document",
                                        value=doc_names[0] if doc_names else None
                                    )
                                    view_btn = gr.Button("View Document", variant="primary")
                                else:
                                    gr.Markdown("⚠️ No documents found")
                                    doc_selector = gr.Radio(choices=[])
                                    view_btn = gr.Button("View Document", interactive=False)
                            
                            with gr.Column(scale=2):
                                doc_viewer = gr.Markdown(value=load_documents_list())
                
                # About page
                about_section = gr.Column(visible=False)
                with about_section:
                    gr.HTML("""
                    <div class='page-header'>
                        <h2>About UroGPT</h2>
                        <p>AI-Powered Urinalysis Assistant</p>
                    </div>
                    """)
                    
                    with gr.Column(elem_classes="content-area"):
                        gr.Markdown("""
                        # About UroGPT
                        
                        **Version:** 2.0 (Sidebar Design)  
                        **Purpose:** AI-assisted urinalysis interpretation and UTI diagnosis
                        
                        ## Features:
                        - 🤖 **AI Chat Assistant** - Ask medical questions
                        - 📷 **Image Analysis** - Analyze urinalysis strip images
                        - ⌨️ **Manual Input** - Enter test values manually
                        - 📚 **Knowledge Base** - Evidence-based medical documents
                        
                        ## Technology Stack:
                        - **LLM:** GPT-4 (OpenAI)
                        - **RAG:** LangChain + FAISS
                        - **Interface:** Gradio
                        
                        ## Disclaimer:
                        ⚠️ **For research and educational purposes only.**  
                        NOT for clinical diagnosis. Always consult healthcare professionals.
                        """)
        
        # Navigation handlers
        def show_chat():
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        def show_image():
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        def show_manual():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        def show_docs():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False)
            )
        
        def show_about():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True)
            )
        
        # Connect navigation
        chat_nav.click(show_chat, outputs=[chat_section, image_section, manual_section, docs_section, about_section])
        image_nav.click(show_image, outputs=[chat_section, image_section, manual_section, docs_section, about_section])
        manual_nav.click(show_manual, outputs=[chat_section, image_section, manual_section, docs_section, about_section])
        docs_nav.click(show_docs, outputs=[chat_section, image_section, manual_section, docs_section, about_section])
        about_nav.click(show_about, outputs=[chat_section, image_section, manual_section, docs_section, about_section])
        
        # Connect functionality
        def respond(message, chat_history):
            if not message or not message.strip():
                return "", chat_history
            updated_history = chat_interface(message, chat_history)
            return "", updated_history
        
        submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        
        analyze_image_btn.click(analyze_from_image, image_input, image_output)
        
        analyze_manual_btn.click(
            analyze_from_values,
            [glucose_input, ph_input, nitrite_input, lymphocyte_input, patient_context_input],
            manual_output
        )
        
        if doc_names:
            view_btn.click(
                lambda doc: view_document(doc, show_summary=False),
                inputs=doc_selector,
                outputs=doc_viewer
            )
    
    return interface


# Main function
def main():
    """Main entry point"""
    initialize_system()
    
    print("="*60)
    print("🌐 Starting web server...")
    print("="*60)
    print()
    print("📱 Open in your browser:")
    print("   Local:   http://localhost:7860")
    print("   Network: http://0.0.0.0:7860")
    print()
    print(" Press Ctrl+C to stop the server")
    print()
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()

