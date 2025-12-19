"""
FastAPI backend for streaming AI agent responses
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import asyncio
from dotenv import load_dotenv
from agent import stream_agent_response
from firebase_client import get_firebase_client

load_dotenv()

app = FastAPI(title="Agentic AI Streaming API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []


class Chat21WebhookRequest(BaseModel):
    """Request model for Chat21 webhook"""
    conversation_id: str
    message_id: str
    user_id: str
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []

@app.get("/")
async def root():
    return {"message": "Agentic AI Streaming API", "status": "running"}

@app.post("/api/chat/stream")
async def stream_chat(request: ChatRequest):
    """
    Stream AI agent responses (direct API endpoint)
    
    Request body:
    {
        "message": "user message",
        "conversation_history": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    if not request.message or not isinstance(request.message, str):
        raise HTTPException(status_code=400, detail="Message is required")
    
    async def generate():
        try:
            async for chunk in stream_agent_response(
                request.message,
                request.conversation_history or []
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


async def process_agent_response(
    conversation_id: str,
    message_id: str,
    user_message: str,
    conversation_history: List[Dict[str, str]],
):
    """
    Background task to process agent response and stream to Firebase
    
    Args:
        conversation_id: Conversation identifier
        message_id: Message identifier
        user_message: User's message
        conversation_history: Previous conversation messages
    """
    firebase = get_firebase_client()
    if not firebase:
        print("Warning: Firebase not configured, cannot write responses")
        return
    
    try:
        full_response = ""
        chunk_count = 0
        
        # Stream agent response
        async for chunk in stream_agent_response(user_message, conversation_history):
            full_response += chunk
            chunk_count += 1
            
            # Write chunk to Firebase
            await firebase.write_response_chunk(
                conversation_id=conversation_id,
                message_id=message_id,
                chunk=chunk,
                is_complete=False,
            )
        
        # Write final complete response
        await firebase.write_complete_response(
            conversation_id=conversation_id,
            message_id=message_id,
            full_response=full_response,
        )
        
        print(f"Successfully processed message {message_id} with {chunk_count} chunks")
    except Exception as e:
        print(f"Error processing agent response: {str(e)}")
        # Write error to Firebase
        if firebase:
            await firebase.write_response_chunk(
                conversation_id=conversation_id,
                message_id=message_id,
                chunk=f"Error: {str(e)}",
                is_complete=True,
            )


@app.post("/api/webhook/chat21")
async def chat21_webhook(
    request: Chat21WebhookRequest, background_tasks: BackgroundTasks
):
    """
    Webhook endpoint for Chat21 Cloud Functions
    
    Receives messages from Chat21 and processes them through the agent.
    Streams responses back to Firebase asynchronously.
    
    Request body:
    {
        "conversation_id": "conv123",
        "message_id": "msg123",
        "user_id": "user123",
        "message": "Hello, how are you?",
        "conversation_history": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    
    Response:
    {
        "status": "accepted",
        "conversation_id": "conv123",
        "message_id": "msg123"
    }
    """
    if not request.message or not isinstance(request.message, str):
        raise HTTPException(status_code=400, detail="Message is required")
    
    if not request.conversation_id or not request.message_id:
        raise HTTPException(
            status_code=400, detail="conversation_id and message_id are required"
        )
    
    # Start background task to process agent response
    background_tasks.add_task(
        process_agent_response,
        request.conversation_id,
        request.message_id,
        request.message,
        request.conversation_history or [],
    )
    
    # Return immediate response to Chat21
    return JSONResponse(
        {
            "status": "accepted",
            "conversation_id": request.conversation_id,
            "message_id": request.message_id,
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

