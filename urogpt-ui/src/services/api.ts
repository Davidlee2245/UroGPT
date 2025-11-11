/**
 * API Service
 * Handles all backend API calls
 */

const API_BASE = 'http://localhost:8000'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface AnalysisRequest {
  glucose?: number
  pH?: number
  nitrite?: number
  lymphocyte?: number
  patient_context?: string
}

export interface AnalysisResponse {
  status: string
  urinalysis_results: Record<string, any>
  report: string
  summary: string
  interpretation: Record<string, string>
  recommendations: string[]
  retrieved_context?: string[]
}

export interface Document {
  filename: string
  filepath: string
  type: string
  size: number
}

/**
 * Send a chat query to the backend
 */
export async function sendChatQuery(query: string): Promise<string> {
  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        patient_context: query
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`API error: ${response.statusText} - ${errorText}`)
    }

    const data = await response.json()
    return data.response || 'No response from server.'
  } catch (error) {
    console.error('Chat API error:', error)
    if (error instanceof Error) {
      return `Error: ${error.message}`
    }
    return `Error: Unable to connect to UroGPT backend. Please ensure the API server is running on ${API_BASE}`
  }
}

/**
 * Analyze urinalysis results
 */
export async function analyzeResults(request: AnalysisRequest): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Analyze image
 */
export async function analyzeImage(file: File, patientContext?: string): Promise<AnalysisResponse> {
  const formData = new FormData()
  formData.append('file', file)
  if (patientContext) {
    formData.append('patient_context', patientContext)
  }

  const response = await fetch(`${API_BASE}/analyze/image`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Get list of documents
 */
export async function getDocuments(): Promise<Document[]> {
  try {
    const response = await fetch(`${API_BASE}/documents`)
    if (!response.ok) {
      throw new Error('Failed to fetch documents')
    }
    const data = await response.json()
    return data.documents || []
  } catch (error) {
    console.error('Failed to fetch documents:', error)
    // Fallback to empty array
    return []
  }
}

/**
 * Get document content
 */
export async function getDocumentContent(filepath: string): Promise<string> {
  try {
    const response = await fetch(`${API_BASE}/documents/content?filepath=${encodeURIComponent(filepath)}`)
    if (!response.ok) {
      throw new Error('Failed to fetch document content')
    }
    const data = await response.json()
    return data.content || 'No content available.'
  } catch (error) {
    console.error('Failed to fetch document content:', error)
    return 'Error loading document content. Please ensure the API server is running.'
  }
}

/**
 * Get cached summary if available
 */
export async function getCachedSummary(filepath: string): Promise<string | null> {
  try {
    const response = await fetch(`${API_BASE}/documents/summary?filepath=${encodeURIComponent(filepath)}`)
    if (!response.ok) {
      return null
    }
    const data = await response.json()
    return data.cached ? data.summary : null
  } catch (error) {
    console.error('Failed to fetch cached summary:', error)
    return null
  }
}

/**
 * Save summary to cache
 */
export async function saveSummaryToCache(filepath: string, summary: string): Promise<void> {
  try {
    await fetch(`${API_BASE}/documents/summary`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        filepath: filepath,
        summary: summary
      }),
    })
  } catch (error) {
    console.error('Failed to save summary to cache:', error)
  }
}

/**
 * Generate document summary using AI
 */
export async function generateSummary(content: string, filepath: string): Promise<string> {
  try {
    // First check if there's a cached summary
    const cachedSummary = await getCachedSummary(filepath)
    if (cachedSummary) {
      return cachedSummary
    }

    // Generate new summary
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        patient_context: `Please provide a concise summary (3-4 paragraphs) of the following medical document:\n\n${content.substring(0, 4000)}`
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`API error: ${response.statusText} - ${errorText}`)
    }

    const data = await response.json()
    const summary = data.response || 'Unable to generate summary.'
    
    // Cache the summary for future use
    await saveSummaryToCache(filepath, summary)
    
    return summary
  } catch (error) {
    console.error('Summary generation error:', error)
    if (error instanceof Error) {
      return `Error: ${error.message}`
    }
    return 'Error: Unable to generate summary. Please ensure the API server is running.'
  }
}

