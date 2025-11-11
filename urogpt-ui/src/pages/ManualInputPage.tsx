/**
 * Manual Input Page
 * Enter urinalysis test values manually
 */

import { useState } from 'react'

export default function ManualInputPage() {
  const [values, setValues] = useState({
    glucose: '',
    ph: '',
    nitrite: '',
    lymphocyte: '',
    context: '',
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    console.log('Submitting values:', values)
    // TODO: Send to backend API
  }

  return (
    <div className="max-w-4xl mx-auto p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-gray-900 mb-2">Manual Input</h1>
        <p className="text-gray-600">Enter your urinalysis test results manually</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid md:grid-cols-2 gap-6">
          {/* Glucose */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Glucose (mg/dL)
            </label>
            <input
              type="number"
              step="0.1"
              value={values.glucose}
              onChange={(e) => setValues({ ...values, glucose: e.target.value })}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-urogpt-500 focus:border-urogpt-500 outline-none transition"
              placeholder="e.g., 3.1"
            />
            <p className="mt-1 text-xs text-gray-500">Normal: 0-15 mg/dL</p>
          </div>

          {/* pH */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              pH Level
            </label>
            <input
              type="number"
              step="0.1"
              value={values.ph}
              onChange={(e) => setValues({ ...values, ph: e.target.value })}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-urogpt-500 focus:border-urogpt-500 outline-none transition"
              placeholder="e.g., 6.8"
            />
            <p className="mt-1 text-xs text-gray-500">Normal: 4.5-8.0</p>
          </div>

          {/* Nitrite */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Nitrite (mg/dL)
            </label>
            <input
              type="number"
              step="0.1"
              value={values.nitrite}
              onChange={(e) => setValues({ ...values, nitrite: e.target.value })}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-urogpt-500 focus:border-urogpt-500 outline-none transition"
              placeholder="e.g., 0.2"
            />
            <p className="mt-1 text-xs text-gray-500">Normal: Negative (0)</p>
          </div>

          {/* Lymphocytes */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Lymphocytes (cells/μL)
            </label>
            <input
              type="number"
              step="0.1"
              value={values.lymphocyte}
              onChange={(e) => setValues({ ...values, lymphocyte: e.target.value })}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-urogpt-500 focus:border-urogpt-500 outline-none transition"
              placeholder="e.g., 1.4"
            />
            <p className="mt-1 text-xs text-gray-500">Normal: &lt;5 cells/μL</p>
          </div>
        </div>

        {/* Patient Context */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Patient Context (Optional)
          </label>
          <textarea
            value={values.context}
            onChange={(e) => setValues({ ...values, context: e.target.value })}
            rows={4}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-urogpt-500 focus:border-urogpt-500 outline-none transition resize-none"
            placeholder="e.g., 45-year-old female with dysuria and frequency..."
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          className="w-full bg-urogpt-600 hover:bg-urogpt-700 text-white font-medium py-3 px-6 rounded-lg transition-colors focus-ring"
        >
          Generate Report
        </button>
      </form>

      {/* Quick Presets */}
      <div className="mt-8 p-6 bg-gray-50 rounded-xl">
        <h3 className="font-semibold text-gray-900 mb-4">Quick Presets:</h3>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() =>
              setValues({ glucose: '5.0', ph: '6.5', nitrite: '0.0', lymphocyte: '2.0', context: '' })
            }
            className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 text-sm font-medium transition"
          >
            Normal Results
          </button>
          <button
            type="button"
            onClick={() =>
              setValues({
                glucose: '8.0',
                ph: '7.5',
                nitrite: '0.5',
                lymphocyte: '12.0',
                context: 'Patient with dysuria and frequency',
              })
            }
            className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 text-sm font-medium transition"
          >
            Possible UTI
          </button>
          <button
            type="button"
            onClick={() =>
              setValues({
                glucose: '45.0',
                ph: '6.2',
                nitrite: '0.0',
                lymphocyte: '3.0',
                context: 'Known diabetic patient',
              })
            }
            className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 text-sm font-medium transition"
          >
            High Glucose
          </button>
        </div>
      </div>
    </div>
  )
}

