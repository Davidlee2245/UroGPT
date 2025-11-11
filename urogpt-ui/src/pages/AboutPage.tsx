/**
 * About Page
 * Information about UroGPT system
 */

import { Bot, Database, Zap, Shield } from 'lucide-react'

export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-gray-900 mb-2">About UroGPT</h1>
        <p className="text-gray-600">AI-Powered Urinalysis Assistant</p>
      </div>

      {/* Overview */}
      <div className="prose prose-lg mb-12">
        <p className="text-gray-700 leading-relaxed">
          UroGPT is an advanced AI system designed to assist healthcare professionals and patients in
          understanding urinalysis test results. Using state-of-the-art natural language processing and
          medical knowledge retrieval, UroGPT provides accurate, evidence-based interpretations of
          urinary tract health indicators.
        </p>
      </div>

      {/* Features */}
      <div className="grid md:grid-cols-2 gap-6 mb-12">
        <div className="bg-white border border-gray-200 rounded-xl p-6">
          <div className="w-12 h-12 bg-urogpt-100 rounded-lg flex items-center justify-center mb-4">
            <Bot className="w-6 h-6 text-urogpt-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">AI Chat Assistant</h3>
          <p className="text-sm text-gray-600">
            Ask questions about urinalysis parameters, UTI diagnosis, and medical interpretations in
            natural language
          </p>
        </div>

        <div className="bg-white border border-gray-200 rounded-xl p-6">
          <div className="w-12 h-12 bg-urogpt-100 rounded-lg flex items-center justify-center mb-4">
            <Zap className="w-6 h-6 text-urogpt-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Image Analysis</h3>
          <p className="text-sm text-gray-600">
            Upload urinalysis strip images for AI-powered automatic parameter detection and analysis
          </p>
        </div>

        <div className="bg-white border border-gray-200 rounded-xl p-6">
          <div className="w-12 h-12 bg-urogpt-100 rounded-lg flex items-center justify-center mb-4">
            <Database className="w-6 h-6 text-urogpt-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Knowledge Base</h3>
          <p className="text-sm text-gray-600">
            Evidence-based medical documents with RAG (Retrieval-Augmented Generation) for accurate
            responses
          </p>
        </div>

        <div className="bg-white border border-gray-200 rounded-xl p-6">
          <div className="w-12 h-12 bg-urogpt-100 rounded-lg flex items-center justify-center mb-4">
            <Shield className="w-6 h-6 text-urogpt-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Privacy First</h3>
          <p className="text-sm text-gray-600">
            Your data is processed securely. No personal health information is stored without consent
          </p>
        </div>
      </div>

      {/* Technology Stack */}
      <div className="bg-gradient-to-r from-urogpt-50 to-blue-50 rounded-xl p-6 border border-urogpt-200 mb-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Technology Stack</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Backend:</h4>
            <ul className="space-y-1 text-gray-700">
              <li>• Python + FastAPI</li>
              <li>• LangChain + OpenAI</li>
              <li>• FAISS Vector Store</li>
              <li>• PyTorch (Image Analysis)</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Frontend:</h4>
            <ul className="space-y-1 text-gray-700">
              <li>• React + TypeScript</li>
              <li>• Vite</li>
              <li>• Tailwind CSS</li>
              <li>• Lucide Icons</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-r-xl">
        <h3 className="font-semibold text-gray-900 mb-2">⚠️ Medical Disclaimer</h3>
        <p className="text-sm text-gray-700 leading-relaxed">
          UroGPT is designed for <strong>research and educational purposes only</strong>. It is NOT
          intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare
          professionals for medical advice, diagnosis, or treatment. The information provided by UroGPT
          should not replace professional medical judgment.
        </p>
      </div>

      {/* Version */}
      <div className="mt-8 text-center text-sm text-gray-500">
        <p>UroGPT Version 2.0.0 | Built with ❤️ for better healthcare</p>
      </div>
    </div>
  )
}

