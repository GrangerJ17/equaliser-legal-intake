from orchestrator.main_orchestration import ChatOrchestrator
from langchain_openai import ChatOpenAI
from embedding_pipeline.embedder import Embedder
from orchestrator.prompts import EQUALISER_SYSTEM_PROMPT
from flask import Flask, request, jsonify
import secrets
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

sessions = {}


def create_chat_session():
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    assistant_llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")  # Or keep Ollama
    embedder = Embedder()

    client = ChatOrchestrator(
        llm=llm,
        assistant_llm=assistant_llm,
        embedder=embedder,
        system_prompt=EQUALISER_SYSTEM_PROMPT
    )

@app.route("/session", methods=['post'])
def create_session():
    session_id = secrets.token_urlsafe(16)
    sessions[session_id] = create_chat_session()

    return jsonify({
        "session_id": session_id
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    user_message = data.get("message")

    if session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400

    client = sessions[session_id]

    ai_message = client.orchestrate(user_message)

    return jsonify({
        "ai_message": ai_message,
        "complete": client.complete
    })


if __name__ == "__main__":
    app.run(debug=True)