import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import ChatLayout from './components/ChatLayout'
import ChatMessage from './components/ChatMessage'
import ChatInput from './components/ChatInput'
import ModeToggle from './components/ModeToggle'

function App() {
  const [messages, setMessages] = useState([
    { text: "Hello! I am your Injection Molding Assistant. Ask me anything about process basics, scientific molding, or material properties found in the ebook.", sender: "bot" }
  ])
  const [input, setInput] = useState("")
  const [mode, setMode] = useState("short") // short | detailed
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim()) return

    const userMsg = { text: input, sender: "user" }
    setMessages(prev => [...prev, userMsg])
    const currentInput = input
    setInput("")
    setLoading(true)

    try {
      const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
      const response = await axios.post(`${backendUrl}/chat`, {
        message: currentInput,
        mode: mode
      })

      const botMsg = { text: response.data.response, sender: "bot" }
      setMessages(prev => [...prev, botMsg])
    } catch (error) {
      console.error("Error:", error)
      const errorText = error.response
        ? `Error: ${error.response.statusText}`
        : "I'm having trouble connecting to the knowledge base. Please ensure the backend is running."
      setMessages(prev => [...prev, { text: errorText, sender: "bot" }])
    }
    setLoading(false)
  }

  return (
    <ChatLayout>
      <div className="mb-6 flex justify-center">
        <ModeToggle mode={mode} setMode={setMode} />
      </div>

      <div className="space-y-2">
        {messages.map((msg, index) => (
          <ChatMessage key={index} message={msg} />
        ))}
        {loading && (
          <div className="flex justify-start mb-6 message-fade-in">
            <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-none px-4 py-3 shadow-sm">
              <div className="flex gap-1.5 items-center">
                <span className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce"></span>
                <span className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                <span className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                <span className="ml-2 text-xs font-medium text-gray-400">Thinking...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="fixed bottom-0 left-0 right-0 z-20">
        <ChatInput
          input={input}
          setInput={setInput}
          onSend={sendMessage}
          loading={loading}
        />
      </div>
    </ChatLayout>
  )
}

export default App
