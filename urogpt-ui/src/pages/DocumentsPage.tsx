/**
 * Documents Page
 * Browse medical knowledge base documents
 */

import { BookOpen, FileText, Search, X, Loader2, Sparkles } from 'lucide-react'
import { useState, useEffect } from 'react'
import { getDocuments, getDocumentContent, generateSummary, getCachedSummary } from '../services/api'

interface Document {
  filename: string
  filepath: string
  type: string
  size: number
}

const mockDocuments = [
  {
    filename: 'urinalysis_basics.txt',
    filepath: 'documents/sample_docs/urinalysis_basics.txt',
    type: 'text',
    size: 2348,
  },
  {
    filename: 'uti_management.txt',
    filepath: 'documents/sample_docs/uti_management.txt',
    type: 'text',
    size: 1876,
  },
]

export default function DocumentsPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [documents, setDocuments] = useState<Document[]>(mockDocuments)
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null)
  const [docContent, setDocContent] = useState('')
  const [docSummary, setDocSummary] = useState('')
  const [isLoadingContent, setIsLoadingContent] = useState(false)
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false)
  const [showSummary, setShowSummary] = useState(false)

  useEffect(() => {
    loadDocuments()
  }, [])

  const loadDocuments = async () => {
    try {
      const docs = await getDocuments()
      if (docs.length > 0) {
        setDocuments(docs)
      }
    } catch (error) {
      console.error('Failed to load documents:', error)
    }
  }

  const handleDocumentClick = async (doc: Document) => {
    setSelectedDoc(doc.filename)
    setIsLoadingContent(true)
    
    try {
      // Load document content and check for cached summary in parallel
      const [content, cachedSummary] = await Promise.all([
        getDocumentContent(doc.filepath),
        getCachedSummary(doc.filepath)
      ])
      
      setDocContent(content)
      
      // If there's a cached summary, show it automatically
      if (cachedSummary) {
        setDocSummary(cachedSummary)
        setShowSummary(true)
      } else {
        // No cached summary, reset summary state
        setDocSummary('')
        setShowSummary(false)
      }
    } catch (error) {
      setDocContent('Error loading document content.')
      setDocSummary('')
      setShowSummary(false)
    } finally {
      setIsLoadingContent(false)
    }
  }

  const handleGenerateSummary = async () => {
    if (!docContent || !selectedDoc) return
    
    setIsGeneratingSummary(true)
    try {
      // Find the selected document to get its filepath
      const doc = documents.find(d => d.filename === selectedDoc)
      if (!doc) {
        throw new Error('Document not found')
      }
      
      const summary = await generateSummary(docContent, doc.filepath)
      setDocSummary(summary)
      setShowSummary(true)
    } catch (error) {
      setDocSummary('Error generating summary.')
    } finally {
      setIsGeneratingSummary(false)
    }
  }

  const handleCloseViewer = () => {
    setSelectedDoc(null)
    setDocContent('')
    setDocSummary('')
    setShowSummary(false)
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="max-w-6xl mx-auto p-8 relative">
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-gray-900 mb-2">Medical Knowledge Base</h1>
        <p className="text-gray-600">Browse medical documents used for evidence-based interpretation</p>
      </div>

      {/* Search */}
      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search documents..."
            className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-urogpt-500 focus:border-urogpt-500 outline-none transition"
          />
        </div>
      </div>

      {/* Documents Grid */}
      <div className="grid gap-4">
        {documents.map((doc, index) => (
          <div
            key={index}
            onClick={() => handleDocumentClick(doc)}
            className="bg-white border border-gray-200 rounded-xl p-6 hover:border-urogpt-500 hover:shadow-md transition-all cursor-pointer"
          >
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-urogpt-100 rounded-lg flex items-center justify-center">
                {doc.type === 'pdf' ? (
                  <FileText className="w-6 h-6 text-urogpt-600" />
                ) : (
                  <BookOpen className="w-6 h-6 text-urogpt-600" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-lg font-semibold text-gray-900 mb-1">{doc.filename}</h3>
                <p className="text-sm text-gray-600 mb-2">Click to view document content</p>
                <div className="flex items-center gap-4 text-xs text-gray-500">
                  <span className="font-medium">{doc.type.toUpperCase()}</span>
                  <span>•</span>
                  <span>{formatFileSize(doc.size)}</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Document Viewer Modal */}
      {selectedDoc && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <div>
                <h2 className="text-xl font-semibold text-gray-900">{selectedDoc}</h2>
                <p className="text-sm text-gray-600 mt-1">Document Viewer</p>
              </div>
              <div className="flex items-center gap-2">
                {docContent && (
                  <>
                    {!showSummary ? (
                      <button
                        onClick={handleGenerateSummary}
                        disabled={isGeneratingSummary}
                        className="flex items-center gap-2 px-4 py-2 bg-urogpt-600 text-white rounded-lg hover:bg-urogpt-700 disabled:opacity-50 transition-colors"
                      >
                        {isGeneratingSummary ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Generating...
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-4 h-4" />
                            {docSummary ? 'Regenerate Summary' : 'Generate Summary'}
                          </>
                        )}
                      </button>
                    ) : (
                      <button
                        onClick={() => setShowSummary(false)}
                        className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                      >
                        Hide Summary
                      </button>
                    )}
                  </>
                )}
                <button
                  onClick={handleCloseViewer}
                  className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-gray-600" />
                </button>
              </div>
            </div>
            
            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
              {isLoadingContent ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 animate-spin text-urogpt-600" />
                </div>
              ) : (
                <>
                  {showSummary && docSummary && (
                    <div className="mb-6 p-4 bg-urogpt-50 border border-urogpt-200 rounded-lg">
                      <h3 className="text-lg font-semibold text-urogpt-900 mb-2 flex items-center gap-2">
                        <Sparkles className="w-5 h-5" />
                        AI Summary
                      </h3>
                      <p className="text-gray-700 whitespace-pre-wrap">{docSummary}</p>
                    </div>
                  )}
                  <div className="prose max-w-none">
                    <pre className="whitespace-pre-wrap font-sans text-sm text-gray-700 leading-relaxed">
                      {docContent}
                    </pre>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="mt-8 p-6 bg-gradient-to-r from-urogpt-50 to-blue-50 rounded-xl border border-urogpt-200">
        <h3 className="font-semibold text-gray-900 mb-2">Corpus Statistics</h3>
        <div className="grid grid-cols-3 gap-4 mt-4">
          <div>
            <div className="text-2xl font-bold text-urogpt-600">{documents.length}</div>
            <div className="text-sm text-gray-600">Total Documents</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-urogpt-600">69</div>
            <div className="text-sm text-gray-600">Knowledge Chunks</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-urogpt-600">15</div>
            <div className="text-sm text-gray-600">PDF Pages</div>
          </div>
        </div>
      </div>
    </div>
  )
}

