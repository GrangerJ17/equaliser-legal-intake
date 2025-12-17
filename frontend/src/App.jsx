import { useEffect, useState } from "react";
import { createSession, sendMessage } from "./api/chatApi";

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [complete, setComplete] = useState(false);
  const [loading, setLoading] = useState(false);

  // Create session on load
  useEffect(() => {
    createSession()
      .then((data) => {
        setSessionId(data.session_id);
      })
      .catch(console.error);
  }, []);

  async function handleSend() {
    if (!input || !sessionId || loading || complete) return;

    setLoading(true);

    // Add user message
    setMessages((m) => [...m, { role: "user", content: input }]);

    const userText = input;
    setInput("");

    try {
      const res = await sendMessage(sessionId, userText);

      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.ai_message },
      ]);

      setComplete(res.complete);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 700, margin: "40px auto" }}>
      <h2>Equaliser Chat</h2>

      <div style={{ border: "1px solid #ccc", padding: 16, minHeight: 300 }}>
        {messages.map((m, i) => (
          <div key={i}>
            <strong>{m.role}:</strong> {m.content}
          </div>
        ))}
        {loading && <em>AI is thinking…</em>}
      </div>

      <div style={{ marginTop: 12 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading || complete}
          style={{ width: "80%" }}
        />
        <button onClick={handleSend} disabled={loading || complete}>
          Send
        </button>
      </div>

      {complete && <p> Chat completed</p>}
    </div>
  );
}

export default App;
