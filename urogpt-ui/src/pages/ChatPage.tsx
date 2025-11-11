/**
 * Chat Page
 * Full-screen chat interface with message history
 */

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2 } from 'lucide-react'
import { sendChatQuery } from '../services/api'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface ChatPageProps {
  initialQuery?: string
}

export default function ChatPage({ initialQuery }: ChatPageProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const hasProcessedInitialQuery = useRef(false)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Handle initial query when component mounts
  useEffect(() => {
    if (initialQuery && !hasProcessedInitialQuery.current) {
      hasProcessedInitialQuery.current = true
      handleQuery(initialQuery)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialQuery])

  const handleQuery = async (query: string) => {
    if (!query.trim() || isLoading) return

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: query }])
    setIsLoading(true)

    try {
      // Get AI response
      const response = await sendChatQuery(query)
      setMessages(prev => [...prev, { role: 'assistant', content: response }])
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.' 
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const userMessage = input.trim()
    if (!userMessage) return
    
    setInput('')
    await handleQuery(userMessage)
  }

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-900">Chat with UroGPT</h1>
        <p className="text-sm text-gray-600">Ask questions about urinalysis and get AI-powered answers</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-20">
            <p className="text-lg mb-2">Start a conversation</p>
            <p className="text-sm">Ask me anything about urinalysis interpretation!</p>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl rounded-2xl px-6 py-4 ${
                  message.role === 'user'
                    ? 'bg-urogpt-600 text-white'
                    : 'bg-white border border-gray-200 text-gray-900'
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
              </div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-3xl rounded-2xl px-6 py-4 bg-white border border-gray-200">
              <Loader2 className="w-5 h-5 animate-spin text-urogpt-600" />
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 bg-white p-6">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask UroGPT a question..."
              className="flex-1 px-6 py-4 border border-gray-300 rounded-full focus:ring-2 focus:ring-urogpt-500 focus:border-urogpt-500 outline-none transition"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="px-8 py-4 bg-urogpt-600 text-white rounded-full hover:bg-urogpt-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 font-medium"
            >
              <Send className="w-5 h-5" />
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

