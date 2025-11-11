/**
 * ModelSelector Component
 * Dropdown for selecting AI model version
 */

import { useState, useRef, useEffect } from 'react'
import { ChevronDown } from 'lucide-react'

const models = [
  { id: 'gpt-4', name: 'GPT-4', description: 'Most capable' },
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', description: 'Fast & powerful' },
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', description: 'Fast responses' },
]

export default function ModelSelector() {
  const [isOpen, setIsOpen] = useState(false)
  const [selected, setSelected] = useState(models[0])
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen])

  return (
    <div className="relative" ref={menuRef}>
      <button
        type="button"
        className="flex items-center gap-1 px-3 py-1.5 hover:bg-gray-100 rounded-lg transition-colors focus-ring text-sm font-medium text-gray-700"
        onClick={() => setIsOpen(!isOpen)}
        aria-label={`Current model: ${selected.name}`}
        aria-haspopup="true"
        aria-expanded={isOpen}
      >
        <span>{selected.name}</span>
        <ChevronDown className="w-4 h-4" />
      </button>

      {isOpen && (
        <div
          className="absolute right-0 mt-2 w-56 bg-white border border-gray-200 rounded-xl shadow-lg overflow-hidden z-20"
          role="menu"
        >
          {models.map((model) => (
            <button
              key={model.id}
              type="button"
              className={`
                w-full px-4 py-3 text-left transition-colors focus-ring
                ${
                  model.id === selected.id
                    ? 'bg-urogpt-50'
                    : 'hover:bg-gray-50'
                }
              `}
              onClick={() => {
                setSelected(model)
                setIsOpen(false)
                console.log('Model selected:', model.name)
              }}
              role="menuitem"
            >
              <div className="font-medium text-gray-800">{model.name}</div>
              <div className="text-xs text-gray-500">{model.description}</div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

