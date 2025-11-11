/**
 * SearchBar Component
 * Main search input with tools, model selector, and mic button
 */

import { useState, useRef, useEffect } from 'react'
import { Plus } from 'lucide-react'
import ToolsMenu from './ToolsMenu'
import ModelSelector from './ModelSelector'

const suggestions = [
  'What does positive nitrite indicate?',
  'Explain normal pH levels in urinalysis',
  'How to interpret elevated protein?',
  'UTI diagnosis criteria',
]

interface SearchBarProps {
  onSubmit?: (query: string) => void
}

export default function SearchBar({ onSubmit }: SearchBarProps) {
  const [query, setQuery] = useState('')
  const [showSuggestions, setShowSuggestions] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Focus on "/" key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '/' && document.activeElement !== inputRef.current) {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      if (onSubmit) {
        onSubmit(query)
      } else {
        console.log('Search:', query)
      }
      setQuery('')
      setShowSuggestions(false)
    }
  }

  return (
    <div className="relative w-full">
      {/* Search form */}
      <form onSubmit={handleSubmit} className="relative">
        <div className="flex items-center gap-3 px-5 py-4 bg-white border-2 border-gray-200 rounded-2xl shadow-sm hover:border-gray-300 focus-within:border-urogpt-500 focus-within:ring-2 focus-within:ring-urogpt-100 transition-all">
          {/* Plus button */}
          <button
            type="button"
            className="flex-shrink-0 p-1 hover:bg-gray-100 rounded-lg transition-colors focus-ring"
            onClick={() => console.log('Plus clicked')}
            aria-label="Add attachment"
          >
            <Plus className="w-5 h-5 text-gray-600" />
          </button>

          {/* Input */}
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setShowSuggestions(true)}
            onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
            placeholder="Ask UroGPT"
            className="flex-1 bg-transparent outline-none text-gray-800 placeholder-gray-400 text-base"
            aria-label="Search UroGPT"
            aria-autocomplete="list"
            aria-controls="search-suggestions"
            aria-expanded={showSuggestions}
          />

          {/* Right controls */}
          <div className="flex items-center gap-2 flex-shrink-0">
            {/* Tools menu */}
            <ToolsMenu />

            {/* Model selector */}
            <ModelSelector />
          </div>
        </div>
      </form>

      {/* Suggestions dropdown */}
      {showSuggestions && query.length === 0 && (
        <div
          id="search-suggestions"
          className="absolute z-10 w-full mt-2 bg-white border border-gray-200 rounded-2xl shadow-lg overflow-hidden"
          role="listbox"
        >
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              type="button"
              className="w-full px-5 py-3 text-left hover:bg-gray-50 transition-colors focus-ring text-sm text-gray-700"
              onClick={() => {
                if (onSubmit) {
                  onSubmit(suggestion)
                }
                setQuery('')
                setShowSuggestions(false)
              }}
              role="option"
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

