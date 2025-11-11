/**
 * Image Analysis Page
 * Upload urinalysis strip images for AI analysis
 */

import { useState } from 'react'
import { Upload, Image as ImageIcon, Loader2, AlertCircle } from 'lucide-react'
import { analyzeImage } from '../services/api'

export default function ImageAnalysisPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<any>(null)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setError(null)
      setResults(null)
      
      // Create preview URL
      const reader = new FileReader()
      reader.onload = (e) => {
        setPreviewUrl(e.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)

    try {
      const data = await analyzeImage(selectedFile)
      setResults(data)
    } catch (err: any) {
      setError(err.message || 'Failed to analyze image')
    } finally {
      setLoading(false)
    }
  }

  const getUTIRiskLevel = (probability: number) => {
    if (probability > 0.7) return { level: 'HIGH', color: 'red' }
    if (probability > 0.4) return { level: 'MODERATE', color: 'yellow' }
    return { level: 'LOW', color: 'green' }
  }

  return (
    <div className="max-w-4xl mx-auto p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-gray-900 mb-2">Image Analysis</h1>
        <p className="text-gray-600">Upload urinalysis strip images for AI-powered analysis</p>
      </div>

      {/* Upload area */}
      <div className="border-2 border-dashed border-gray-300 rounded-2xl p-12 text-center hover:border-urogpt-500 transition-colors">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
          className="hidden"
          id="image-upload"
        />
        <label
          htmlFor="image-upload"
          className="cursor-pointer flex flex-col items-center gap-4"
        >
          <div className="w-16 h-16 bg-urogpt-100 rounded-full flex items-center justify-center">
            <ImageIcon className="w-8 h-8 text-urogpt-600" />
          </div>
          <div>
            <p className="text-lg font-medium text-gray-900 mb-1">
              Click to upload or drag and drop
            </p>
            <p className="text-sm text-gray-500">PNG, JPG up to 10MB</p>
          </div>
        </label>
      </div>

      {/* Image Preview */}
      {previewUrl && (
        <div className="mt-6 bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Preview</h3>
          <div className="flex justify-center">
            <img 
              src={previewUrl} 
              alt="Upload preview" 
              className="max-w-full max-h-96 rounded-lg shadow-lg"
            />
          </div>
          <div className="mt-6 flex gap-3">
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="flex-1 bg-urogpt-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-urogpt-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Upload className="w-5 h-5" />
                  Analyze Image
                </>
              )}
            </button>
            <button
              onClick={() => {
                setSelectedFile(null)
                setPreviewUrl(null)
                setResults(null)
                setError(null)
              }}
              className="px-6 py-3 rounded-lg font-medium border border-gray-300 hover:bg-gray-50 transition-colors"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-6 bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-red-900">Error</h4>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="mt-6 bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Analysis Results</h3>
          
          {/* UTI Risk */}
          {results.urinalysis_results?.UTI_probability !== undefined && (
            <div className={`mb-6 p-4 rounded-lg ${
              getUTIRiskLevel(results.urinalysis_results.UTI_probability).color === 'red' 
                ? 'bg-red-50 border border-red-200' 
                : getUTIRiskLevel(results.urinalysis_results.UTI_probability).color === 'yellow'
                ? 'bg-yellow-50 border border-yellow-200'
                : 'bg-green-50 border border-green-200'
            }`}>
              <div className="flex items-center justify-between">
                <span className="font-semibold text-gray-900">UTI Risk:</span>
                <span className={`font-bold ${
                  getUTIRiskLevel(results.urinalysis_results.UTI_probability).color === 'red'
                    ? 'text-red-700'
                    : getUTIRiskLevel(results.urinalysis_results.UTI_probability).color === 'yellow'
                    ? 'text-yellow-700'
                    : 'text-green-700'
                }`}>
                  {getUTIRiskLevel(results.urinalysis_results.UTI_probability).level} 
                  ({(results.urinalysis_results.UTI_probability * 100).toFixed(1)}%)
                </span>
              </div>
            </div>
          )}

          {/* Parameters */}
          <div className="space-y-3">
            {[
              { label: 'Confidence', value: `${((results.urinalysis_results?.confidence || 0) * 100).toFixed(1)}%` },
              { label: 'Main Class', value: results.urinalysis_results?.main_class || 'N/A' },
              { label: 'Glucose', value: `${results.urinalysis_results?.glucose || 0} mg/dL` },
              { label: 'pH', value: results.urinalysis_results?.pH || 'N/A' },
              { label: 'Nitrite', value: `${results.urinalysis_results?.nitrite || 0} mg/dL` },
              { label: 'Protein', value: `${results.urinalysis_results?.protein || 0} mg/dL` },
              { label: 'Hemoglobin', value: `${results.urinalysis_results?.hemoglobin || 0} mg/dL` },
              { label: 'Bilirubin', value: `${results.urinalysis_results?.bilirubin || 0} mg/dL` },
            ].map((item, idx) => (
              <div key={idx} className="flex justify-between py-2 border-b border-gray-100 last:border-0">
                <span className="text-gray-600">{item.label}</span>
                <span className="font-semibold text-gray-900">{item.value}</span>
              </div>
            ))}
          </div>

          {/* Report Summary */}
          {results.summary && (
            <div className="mt-6 pt-6 border-t border-gray-200">
              <h4 className="font-semibold text-gray-900 mb-2">Summary</h4>
              <p className="text-sm text-gray-700">{results.summary}</p>
            </div>
          )}
        </div>
      )}

      {/* Info */}
      {!results && (
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-xl p-6">
          <h3 className="font-semibold text-gray-900 mb-2">How it works:</h3>
          <ul className="space-y-2 text-sm text-gray-700">
            <li>• Upload a clear photo of your urinalysis test strip</li>
            <li>• AI analyzes glucose, pH, nitrite, and other parameters</li>
            <li>• Get instant results with medical interpretation</li>
            <li>• Download report or share with your doctor</li>
          </ul>
        </div>
      )}
    </div>
  )
}

