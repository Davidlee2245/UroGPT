/**
 * Sidebar Component
 * Left navigation panel with new chat, recent items, and settings
 */

import { PenSquare, Clock, X } from 'lucide-react'
import IconButton from './IconButton'
import RecentItem from './RecentItem'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  currentPage: string
  onNavigate: (page: string) => void
}

// Recent chat history (empty for now - will be populated from backend)
const recentChats: string[] = []

export default function Sidebar({ isOpen, onClose, currentPage, onNavigate }: SidebarProps) {
  return (
    <>
      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-40
          w-64 bg-gray-50 border-r border-gray-200
          transform transition-transform duration-300 ease-in-out
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          flex flex-col
        `}
        aria-label="Main navigation"
      >
        {/* Close button (mobile only) */}
        <div className="lg:hidden flex justify-end p-4">
          <IconButton
            icon={<X className="w-5 h-5" />}
            label="Close sidebar"
            onClick={onClose}
          />
        </div>

        {/* Top section */}
        <div className="p-4 space-y-2">
          <button
            className="w-full flex items-center gap-3 px-4 py-3 rounded-2xl bg-white hover:bg-gray-100 transition-colors focus-ring text-left border border-gray-200"
            onClick={() => onNavigate('home')}
          >
            <PenSquare className="w-5 h-5 text-gray-600" />
            <span className="font-medium text-gray-700">New chat</span>
          </button>

          {/* Navigation buttons */}
          <button
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl transition-colors focus-ring text-left ${
              currentPage === 'image' ? 'bg-gray-200' : 'hover:bg-gray-100'
            }`}
            onClick={() => onNavigate('image')}
          >
            <span className="font-medium text-gray-700">Image Analysis</span>
          </button>

          <button
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl transition-colors focus-ring text-left ${
              currentPage === 'manual' ? 'bg-gray-200' : 'hover:bg-gray-100'
            }`}
            onClick={() => onNavigate('manual')}
          >
            <span className="font-medium text-gray-700">Manual Input</span>
          </button>

          <button
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl transition-colors focus-ring text-left ${
              currentPage === 'docs' ? 'bg-gray-200' : 'hover:bg-gray-100'
            }`}
            onClick={() => onNavigate('docs')}
          >
            <span className="font-medium text-gray-700">Documents</span>
          </button>

          <button
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl transition-colors focus-ring text-left ${
              currentPage === 'about' ? 'bg-gray-200' : 'hover:bg-gray-100'
            }`}
            onClick={() => onNavigate('about')}
          >
            <span className="font-medium text-gray-700">About</span>
          </button>
        </div>

        {/* Recent section - hidden when empty */}
        {recentChats.length > 0 && (
          <div className="flex-1 overflow-y-auto px-4">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 px-2">
              Recent
            </h2>
            <div className="space-y-1">
              {recentChats.map((chat, index) => (
                <RecentItem
                  key={index}
                  title={chat}
                  onClick={() => console.log('Open:', chat)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Bottom section */}
        <div className="p-4 border-t border-gray-200 space-y-2">
          <button
            className="w-full flex items-center gap-3 px-4 py-3 rounded-2xl hover:bg-gray-100 transition-colors focus-ring text-left"
            onClick={() => console.log('Activity')}
          >
            <Clock className="w-5 h-5 text-gray-600" />
            <span className="text-sm text-gray-700">Activity</span>
          </button>
        </div>
      </aside>
    </>
  )
}

