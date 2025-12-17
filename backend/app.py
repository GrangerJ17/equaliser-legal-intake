from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import secrets

from orchestrator.main_orchestration import ChatOrchestrator
from langchain_openai import ChatOpenAI
from embedding_pipeline.embedder import Embedder
from orchestrator.prompts import EQUALISER_SYSTEM_PROMPT

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions: dict[str, ChatOrchestrator] = {}



class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    ai_message: str
    complete: bool



def create_chat_session() -> ChatOrchestrator:
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    assistant_llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    embedder = Embedder()

    return ChatOrchestrator(
        llm=llm,
        assistant_llm=assistant_llm,
        embedder=embedder,
        system_prompt=EQUALISER_SYSTEM_PROMPT,
    )



@app.post("/session", response_model=SessionResponse)
async def create_session():
    session_id = secrets.token_urlsafe(16)
    sessions[session_id] = create_chat_session()
    return {"session_id": session_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    client = sessions.get(request.session_id)

    if not client:
        raise HTTPException(status_code=400, detail="Invalid session")

    # orchestrate is sync → safe to call directly
    ai_message = client.orchestrate(request.message)

    return {
        "ai_message": ai_message,
        "complete": client.complete,
    }
