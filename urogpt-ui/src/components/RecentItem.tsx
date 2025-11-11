/**
 * RecentItem Component
 * Single recent chat item in sidebar
 */

import { MessageSquare } from 'lucide-react'

interface RecentItemProps {
  title: string
  onClick: () => void
}

export default function RecentItem({ title, onClick }: RecentItemProps) {
  return (
    <button
      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-gray-100 transition-colors focus-ring text-left group"
      onClick={onClick}
    >
      <MessageSquare className="w-4 h-4 text-gray-400 flex-shrink-0 group-hover:text-gray-600" />
      <span className="text-sm text-gray-700 truncate group-hover:text-gray-900">
        {title}
      </span>
    </button>
  )
}

