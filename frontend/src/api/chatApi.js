const API_BASE = "http://localhost:5000";

export async function createSession() {
  const res = await fetch(`${API_BASE}/session`, {
    method: "POST",
  });

  if (!res.ok) {
    throw new Error("Failed to create session");
  }

  return res.json(); // { session_id }
}

export async function sendMessage(sessionId, message) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      message,
    }),
  });

  if (!res.ok) {
    throw new Error("Chat request failed");
  }

  return res.json(); // { ai_message, complete }
}