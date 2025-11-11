/**
 * MainContent Component
 * Central area with greeting, search bar, and action chips
 */

import SearchBar from './SearchBar'
import ActionChips from './ActionChips'
import Banner from './Banner'

interface MainContentProps {
  onNavigate: (page: string, query?: string) => void
}

export default function MainContent({ onNavigate }: MainContentProps) {
  const handleSearch = (query: string) => {
    // Navigate to chat page and pass the query
    onNavigate('chat', query)
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-6 lg:p-12">
      <div className="w-full max-w-3xl space-y-8">
        {/* Greeting */}
        <h1 className="text-4xl lg:text-5xl font-normal text-center text-gray-800 mb-12">
          Hello, <span className="text-urogpt-600 font-medium">UroGPT</span>
        </h1>

        {/* Search bar */}
        <SearchBar onSubmit={handleSearch} />

        {/* Action chips */}
        <ActionChips onNavigate={onNavigate} />

        {/* Learn button */}
        <div className="flex justify-center mt-6">
          <button
            className="px-6 py-2 rounded-full bg-gray-100 hover:bg-gray-200 text-sm font-medium text-gray-700 transition-colors focus-ring"
            onClick={() => onNavigate('about')}
          >
            Learn
          </button>
        </div>
      </div>

      {/* Bottom banner */}
      <Banner />
    </div>
  )
}

