/**
 * ActionChips Component
 * Row of action buttons below search bar
 */

import { Image, FileText, BookOpen } from 'lucide-react'
import Chip from './Chip'

const actions = [
  { icon: <Image className="w-5 h-5" />, label: 'Image Analysis', page: 'image' },
  { icon: <FileText className="w-5 h-5" />, label: 'Manual Input', page: 'manual' },
  { icon: <BookOpen className="w-5 h-5" />, label: 'Documents', page: 'docs' },
]

interface ActionChipsProps {
  onNavigate: (page: string) => void
}

export default function ActionChips({ onNavigate }: ActionChipsProps) {
  return (
    <div className="flex flex-wrap gap-3 justify-center">
      {actions.map((action) => (
        <Chip
          key={action.label}
          icon={action.icon}
          label={action.label}
          onClick={() => onNavigate(action.page)}
        />
      ))}
    </div>
  )
}

