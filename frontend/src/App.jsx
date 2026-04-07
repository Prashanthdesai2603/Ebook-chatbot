import { useState, useRef, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import axios from 'axios'
import { AlertCircle, Terminal } from 'lucide-react'
import ChatLayout from './components/ChatLayout'
import ChatMessage from './components/ChatMessage'
import ChatInput from './components/ChatInput'
import Login from './pages/Login'
import SignIn from './pages/SignIn'
import ProtectedRoute from './components/Auth/ProtectedRoute'

function Chatbot() {
  const initialMessages = [
    { text: "Hello! I am your Injection Molding Assistant. Ask me anything about process basics, scientific molding, or material properties found in the ebook.", sender: "bot" }
  ]
  const [messages, setMessages] = useState(initialMessages)
  const [input, setInput] = useState("")
  const [mode, setMode] = useState("short") // short | detailed
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [username, setUsername] = useState(localStorage.getItem('username') || 'Professional User')
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, loading])

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    window.location.href = '/';
  };

  const handleClearChat = () => {
    if (window.confirm("Are you sure you want to clear your chat history?")) {
      setMessages(initialMessages);
      setError(null);
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return

    const userMsg = { text: input, sender: "user" }
    setMessages(prev => [...prev, userMsg])
    const currentInput = input
    setInput("")
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post("http://localhost:8000/api/chat", {
        message: currentInput,
        mode: mode
      })

      const botMsg = { text: response.data.response, sender: "bot" }
      setMessages(prev => [...prev, botMsg])
    } catch (error) {
      console.error("Error:", error)
      setError("I'm having trouble connecting to the knowledge base. Please ensure the backend is running.")
    }
    setLoading(false)
  }

  return (
    <ChatLayout onLogout={handleLogout} onClearChat={handleClearChat} username={username}>
      <div className="flex flex-col gap-2 min-h-full">
        {/* Message List */}
        <div className="flex flex-col gap-3">
          {messages.map((msg, index) => (
            <ChatMessage key={index} message={msg} />
          ))}
          
          {loading && (
            <div className="flex justify-start px-2 md:px-0 message-fade-in">
              <div className="bg-gray-100/50 backdrop-blur-sm border border-gray-100 rounded-2xl rounded-tl-sm px-5 py-3 shadow-sm ring-1 ring-black/5">
                <div className="flex gap-2 items-center">
                  <div className="flex gap-1">
                    <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                    <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                    <span className="w-1.5 h-1.5 bg-blue-600 rounded-full animate-bounce"></span>
                  </div>
                  <span className="ml-2 text-xs font-bold text-blue-600 uppercase tracking-wider">AI is typing...</span>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="flex justify-center px-4 my-4 message-fade-in">
              <div className="bg-red-50 border border-red-100 rounded-xl px-4 py-3 flex items-center gap-3 text-red-700 shadow-sm shadow-red-100">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <p className="text-sm font-semibold">{error}</p>
                <button 
                  onClick={() => setError(null)}
                  className="ml-auto text-[10px] font-black uppercase underline decoration-2 underline-offset-4 hover:text-red-900"
                >
                  Dismiss
                </button>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} className="h-4" />
        </div>
      </div>

      {/* Floating Input area */}
      <div className="sticky bottom-0 left-0 right-0 z-30">
        <ChatInput
          input={input}
          setInput={setInput}
          onSend={sendMessage}
          loading={loading}
          mode={mode}
          setMode={setMode}
        />
      </div>
    </ChatLayout>
  )
}

function App() {
  const token = localStorage.getItem('token');
  const isAuthenticated = token && token !== 'undefined' && token !== 'null';

  return (
    <Router>
      <Routes>
        <Route 
          path="/" 
          element={isAuthenticated ? <Navigate to="/chat" replace /> : <Login />} 
        />
        <Route path="/signin" element={<SignIn />} />
        <Route
          path="/chat"
          element={
            <ProtectedRoute>
              <Chatbot />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  )
}

export default App

