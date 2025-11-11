/**
 * IconButton Component
 * Reusable icon-only button
 */

interface IconButtonProps {
  icon: React.ReactNode
  label: string
  onClick: () => void
  variant?: 'default' | 'primary'
}

export default function IconButton({
  icon,
  label,
  onClick,
  variant = 'default',
}: IconButtonProps) {
  const baseClasses = 'p-2 rounded-lg transition-colors focus-ring'
  const variantClasses = {
    default: 'hover:bg-gray-100 text-gray-600',
    primary: 'bg-urogpt-600 hover:bg-urogpt-700 text-white',
  }

  return (
    <button
      className={`${baseClasses} ${variantClasses[variant]}`}
      onClick={onClick}
      aria-label={label}
    >
      {icon}
    </button>
  )
}

