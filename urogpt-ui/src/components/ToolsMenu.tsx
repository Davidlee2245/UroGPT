/**
 * ToolsMenu Component
 * Accessible popover menu for tools selection
 */

import { useState, useRef, useEffect } from 'react'
import { Wrench } from 'lucide-react'

const tools = [
  'Code interpreter',
  'Web search',
  'Image generation',
  'Data analysis',
  'File upload',
]

export default function ToolsMenu() {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const menuRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)

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

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen) return

    switch (e.key) {
      case 'Escape':
        setIsOpen(false)
        buttonRef.current?.focus()
        break
      case 'ArrowDown':
        e.preventDefault()
        setSelectedIndex((prev) => (prev + 1) % tools.length)
        break
      case 'ArrowUp':
        e.preventDefault()
        setSelectedIndex((prev) => (prev - 1 + tools.length) % tools.length)
        break
      case 'Enter':
      case ' ':
        e.preventDefault()
        console.log('Selected tool:', tools[selectedIndex])
        setIsOpen(false)
        break
    }
  }

  return (
    <div className="relative" ref={menuRef}>
      <button
        ref={buttonRef}
        type="button"
        className="p-2 hover:bg-gray-100 rounded-lg transition-colors focus-ring"
        onClick={() => setIsOpen(!isOpen)}
        onKeyDown={handleKeyDown}
        aria-label="Tools"
        aria-haspopup="true"
        aria-expanded={isOpen}
      >
        <Wrench className="w-4 h-4 text-gray-600" />
      </button>

      {isOpen && (
        <div
          className="absolute right-0 mt-2 w-48 bg-white border border-gray-200 rounded-xl shadow-lg overflow-hidden z-20"
          role="menu"
          aria-orientation="vertical"
        >
          {tools.map((tool, index) => (
            <button
              key={tool}
              type="button"
              className={`
                w-full px-4 py-2 text-left text-sm transition-colors focus-ring
                ${
                  index === selectedIndex
                    ? 'bg-urogpt-50 text-urogpt-700'
                    : 'text-gray-700 hover:bg-gray-50'
                }
              `}
              onClick={() => {
                console.log('Tool selected:', tool)
                setIsOpen(false)
              }}
              onMouseEnter={() => setSelectedIndex(index)}
              role="menuitem"
            >
              {tool}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

