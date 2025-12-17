import { useEffect, useState, useRef } from "react";
import { createSession, sendMessage } from "./api/chatApi";
import "./App.css";

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [complete, setComplete] = useState(false);
  const [loading, setLoading] = useState(false);
  
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Create session on load
  useEffect(() => {
    createSession()
      .then((data) => {
        setSessionId(data.session_id);
        // Add welcome message
        setMessages([{
          role: "assistant",
          content: "Hello, I'm Equaliser. I'm here to help understand your legal situation. What brings you here today?"
        }]);
      })
      .catch(console.error);
  }, []);

  async function handleSend() {
    if (!input.trim() || !sessionId || loading || complete) return;

    const userText = input.trim();
    setInput("");
    setLoading(true);

    // Add user message immediately
    setMessages((m) => [...m, { role: "user", content: userText }]);

    try {
      const res = await sendMessage(sessionId, userText);

      // Add AI response
      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.ai_message },
      ]);

      setComplete(res.complete);
    } catch (err) {
      console.error(err);
      setMessages((m) => [
        ...m,
        { role: "error", content: "Sorry, something went wrong. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-container">
      <div className="chat-wrapper">
        
        {/* Header */}
        <div className="chat-header">
          <div className="header-content">
            <h1>Equaliser</h1>
            <p>Confidential Legal Intake Assistant</p>
          </div>
          {sessionId && (
            <div className="session-indicator">
              <span className="status-dot"></span>
              <span>Connected</span>
            </div>
          )}
        </div>

        {/* Messages */}
        <div className="messages-container">
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>
              <div className="message-content">
                {m.content}
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="message assistant">
              <div className="message-content typing">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="input-container">
          {complete ? (
            <div className="completion-message">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <path d="M20 6L9 17L4 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <span>Chat completed. A specialist will contact you soon.</span>
            </div>
          ) : (
            <div className="input-wrapper">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Type your message... (Press Enter to send)"
                disabled={loading || complete}
                rows={1}
              />
              <button 
                onClick={handleSend} 
                disabled={loading || complete || !input.trim()}
                className="send-button"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

export default App;