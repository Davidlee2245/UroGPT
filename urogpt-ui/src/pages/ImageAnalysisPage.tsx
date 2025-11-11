/**
 * Image Analysis Page
 * Upload urinalysis strip images for AI analysis
 */

import { Upload, Image as ImageIcon } from 'lucide-react'

export default function ImageAnalysisPage() {
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      console.log('Image uploaded:', file.name)
      // TODO: Send to backend API
    }
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

      {/* Info */}
      <div className="mt-8 bg-blue-50 border border-blue-200 rounded-xl p-6">
        <h3 className="font-semibold text-gray-900 mb-2">How it works:</h3>
        <ul className="space-y-2 text-sm text-gray-700">
          <li>• Upload a clear photo of your urinalysis test strip</li>
          <li>• AI analyzes glucose, pH, nitrite, and other parameters</li>
          <li>• Get instant results with medical interpretation</li>
          <li>• Download report or share with your doctor</li>
        </ul>
      </div>
    </div>
  )
}

