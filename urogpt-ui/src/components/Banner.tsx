/**
 * Banner Component
 * Bottom promotional banner
 */

import { X } from 'lucide-react'
import { useState } from 'react'

export default function Banner() {
  const [isVisible, setIsVisible] = useState(true)

  if (!isVisible) return null

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-10 max-w-2xl w-full px-4">
      <div className="bg-white border border-gray-200 rounded-2xl shadow-lg p-4 flex items-center gap-4">
        <button
          className="flex-shrink-0 p-1 hover:bg-gray-100 rounded-lg transition-colors focus-ring"
          onClick={() => setIsVisible(false)}
          aria-label="Dismiss banner"
        >
          <X className="w-5 h-5 text-gray-500" />
        </button>

        <div className="flex-1">
          <p className="text-sm text-gray-700">
            <span className="font-semibold">New!</span> Video generation just got better with Veo 3.1
          </p>
        </div>

        <button
          className="flex-shrink-0 px-6 py-2 bg-urogpt-600 hover:bg-urogpt-700 text-white text-sm font-medium rounded-full transition-colors focus-ring"
          onClick={() => console.log('Try Now clicked')}
        >
          Try Now
        </button>
      </div>
    </div>
  )
}

