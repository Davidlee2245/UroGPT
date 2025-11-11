/**
 * Chip Component
 * Reusable action chip button
 */

interface ChipProps {
  icon: React.ReactNode
  label: string
  onClick: () => void
}

export default function Chip({ icon, label, onClick }: ChipProps) {
  return (
    <button
      className="flex items-center gap-2 px-5 py-3 bg-gray-100 hover:bg-gray-200 rounded-2xl transition-colors focus-ring group"
      onClick={onClick}
      aria-label={label}
    >
      <span className="text-gray-600 group-hover:text-gray-800 transition-colors">
        {icon}
      </span>
      <span className="text-sm font-medium text-gray-700 group-hover:text-gray-900">
        {label}
      </span>
    </button>
  )
}

