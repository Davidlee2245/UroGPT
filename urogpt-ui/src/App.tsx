/**
 * UroGPT - Gemini-style Landing UI
 * Main application component
 */

import { useState } from 'react'
import Sidebar from './components/Sidebar'
import MainContent from './components/MainContent'
import ChatPage from './pages/ChatPage'
import ImageAnalysisPage from './pages/ImageAnalysisPage'
import ManualInputPage from './pages/ManualInputPage'
import DocumentsPage from './pages/DocumentsPage'
import AboutPage from './pages/AboutPage'
import { Menu } from 'lucide-react'

type PageType = 'home' | 'chat' | 'image' | 'manual' | 'docs' | 'about'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [currentPage, setCurrentPage] = useState<PageType>('home')
  const [initialQuery, setInitialQuery] = useState<string>('')

  const handleNavigate = (page: string, query?: string) => {
    setCurrentPage(page as PageType)
    if (query) {
      setInitialQuery(query)
    }
    setSidebarOpen(false) // Close sidebar on mobile after navigation
  }

  return (
    <div className="flex h-screen overflow-hidden bg-white">
      {/* Skip to content link for accessibility */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:p-4 focus:bg-urogpt-500 focus:text-white focus:rounded-md focus:top-4 focus:left-4"
      >
        Skip to content
      </a>

      {/* Mobile hamburger button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 rounded-lg bg-white shadow-md hover:bg-gray-50 focus-ring"
        aria-label="Toggle sidebar"
        aria-expanded={sidebarOpen}
      >
        <Menu className="w-6 h-6 text-gray-700" />
      </button>

      {/* Sidebar */}
      <Sidebar 
        isOpen={sidebarOpen} 
        onClose={() => setSidebarOpen(false)}
        currentPage={currentPage}
        onNavigate={handleNavigate}
      />

      {/* Main content */}
      <main id="main-content" className="flex-1 overflow-y-auto">
        {currentPage === 'home' && <MainContent onNavigate={handleNavigate} />}
        {currentPage === 'chat' && <ChatPage initialQuery={initialQuery} />}
        {currentPage === 'image' && <ImageAnalysisPage />}
        {currentPage === 'manual' && <ManualInputPage />}
        {currentPage === 'docs' && <DocumentsPage />}
        {currentPage === 'about' && <AboutPage />}
      </main>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-30"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}
    </div>
  )
}

export default App

