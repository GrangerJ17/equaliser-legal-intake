from orchestrator.main_orchestration import ChatOrchestrator
from langchain_openai import ChatOpenAI
from embedding_pipeline.embedder import Embedder
from orchestrator.prompts import EQUALISER_SYSTEM_PROMPT
from flask import Flask, request, jsonify
import secrets


# In-memory storage of current session ID
# To extend to persistent db
store = {}


# for testing in CLI

def main():

    uuid = secrets.token_urlsafe(16)
    store[uuid] = None
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    assistant_llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")  # Or keep Ollama
    embedder = Embedder()

    client = ChatOrchestrator(
        llm=llm,
        assistant_llm=assistant_llm,
        embedder=embedder,
        system_prompt=EQUALISER_SYSTEM_PROMPT
    )
    user_message = None
    # Manually call the step that determines the user context
    
    while client.message_count < client.message_limit:
        try:
            if client.complete == False:
                user_message = input("User input:")
                
                

            ai_message = client.orchestrate(user_message)

            if ai_message is None:
                
                continue
            
            print("AI: ", ai_message)

            if client.complete == True:
                break

        except Exception as e:
            print("Error: ", e)

    
            
if __name__ == "__main__":
    import sys
    main()

