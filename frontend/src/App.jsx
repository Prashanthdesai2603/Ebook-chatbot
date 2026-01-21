import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './index.css'

function App() {
  const [messages, setMessages] = useState([
    { text: "Hello! Ask me anything about the ebook.", sender: "bot" }
  ])
  const [input, setInput] = useState("")
  const [mode, setMode] = useState("short") // short | detailed
  const [loading, setLoading] = useState(false)
  const chatEndRef = useRef(null)

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim()) return

    const userMsg = { text: input, sender: "user" }
    setMessages(prev => [...prev, userMsg])
    setInput("")
    setLoading(true)

    try {
      // Logic to handle potential trailing slash or simple URL structure
      const response = await axios.post("http://localhost:8000/chat", {
        message: userMsg.text,
        mode: mode
      })

      const botMsg = { text: response.data.response, sender: "bot" }
      setMessages(prev => [...prev, botMsg])
    } catch (error) {
      console.error("Error:", error)
      const errorText = error.response ? `Error: ${error.response.statusText}` : "Error: Could not reach the librarian. Is the backend running?"
      setMessages(prev => [...prev, { text: errorText, sender: "bot" }])
    }
    setLoading(false)
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="app-container">
      <header className="chat-header">
        <div className="logo-area">
          <h1>📚 eBook Chatbot</h1>
          <p className="subtitle"> Ask questions about your ebook</p>
        </div>
        <div className="mode-toggle">
          <button
            className={`toggle-btn ${mode === 'short' ? 'active' : ''}`}
            onClick={() => setMode('short')}
          >
            Short
          </button>
          <button
            className={`toggle-btn ${mode === 'detailed' ? 'active' : ''}`}
            onClick={() => setMode('detailed')}
          >
            Detailed
          </button>
        </div>
      </header>

      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message-bubble ${msg.sender}`}>
            <div className="sender-label">{msg.sender === 'bot' ? 'AI' : 'You'}</div>
            <div className="message-content">
              {msg.text.split('\n').map((line, i) => (
                <p key={i}>{line}</p>
              ))}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message-bubble bot">
            <div className="typing-indicator">Thinking...</div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <div className="input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Ask a question about the book..."
          rows="1"
        />
        <button onClick={sendMessage} disabled={loading} className="send-btn">
          ➤
        </button>
      </div>
    </div>
  )
}

export default App
