# Python Backend for Agentic AI Streaming

Python FastAPI backend using LangChain and LangGraph for agentic AI responses.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
# Optional: Add Firebase configuration for Chat21 integration
```

## Running the Server

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## Architecture

This agent server supports two integration patterns:

1. **Direct API**: Frontend calls agent server directly (existing pattern)
2. **Webhook Pattern**: Mini program → Firebase → Chat21 → Agent Server (new pattern)

See [ARCHITECTURE.md](../ARCHITECTURE.md) for detailed architecture documentation.

## API Endpoints

### POST /api/webhook/chat21

Webhook endpoint for Chat21 Cloud Functions. Receives messages from Chat21 and processes them through the agent, streaming responses back to Firebase.

**Request Body:**
```json
{
  "conversation_id": "conv123",
  "message_id": "msg123",
  "user_id": "user123",
  "message": "Hello, how are you?",
  "conversation_history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```

**Response:**
```json
{
  "status": "accepted",
  "conversation_id": "conv123",
  "message_id": "msg123"
}
```

The agent processes the message asynchronously and streams chunks to Firebase at:
- `conversations/{conversation_id}/aiResponses/{message_id}`

**Example curl:**
```bash
curl -X POST http://localhost:8000/api/webhook/chat21 \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv123",
    "message_id": "msg123",
    "user_id": "user123",
    "message": "Hello, how are you?"
  }'
```

### POST /api/chat/stream

Stream AI agent responses (direct API endpoint).

**Request Body:**
```json
{
  "message": "Hello, how are you?",
  "conversation_history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```

**Response:** Server-Sent Events (SSE) stream with AI response chunks.

**Example curl commands:**

Simple request (message only):
```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?"
  }'
```

Request with conversation history:
```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What did I ask you before?",
    "conversation_history": [
      {"role": "user", "content": "What is the capital of France?"},
      {"role": "assistant", "content": "The capital of France is Paris."}
    ]
  }'
```

**Note:** The response will be streamed as Server-Sent Events (SSE). Each chunk will appear as `data: <chunk>\n\n` in the output.

## Chat21 Integration

Chat21 is integrated as a Python module (`chat21.py`). Chat21 is built on Firebase Cloud Functions and requires Firebase setup.

**Important:** Chat21 must be deployed to your Firebase project first. See the [Chat21 Cloud Functions repository](https://github.com/chat21/chat21-cloud-functions) for deployment instructions.

### Setup Steps

1. **Deploy Chat21 Cloud Functions to Firebase:**
   ```bash
   git clone https://github.com/chat21/chat21-cloud-functions.git
   cd chat21-cloud-functions
   npm install
   firebase login
   firebase use --add  # Select your Firebase project
   firebase deploy
   ```

2. **Add environment variables to your `.env` file:**
   ```env
   # Firebase configuration for Chat21
   FIREBASE_PROJECT_ID=your_firebase_project_id
   FIREBASE_FUNCTIONS_REGION=us-central1  # Optional, defaults to us-central1
   FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/serviceAccountKey.json  # Optional
   # OR set CHAT21_FUNCTIONS_URL directly:
   # CHAT21_FUNCTIONS_URL=https://us-central1-your-project.cloudfunctions.net
   ```

3. **Get Firebase Service Account Key:**
   - Go to Firebase Console → Project Settings → Service Accounts
   - Click "Generate New Private Key"
   - Save the JSON file and set `FIREBASE_SERVICE_ACCOUNT_PATH` in `.env`

4. **Use the Chat21 client in your code:**
   ```python
   from chat21 import get_chat21_client
   
   client = get_chat21_client()
   if client:
       # Create a custom token for a user
       custom_token = await client.create_custom_token("user123")
       
       # Send a message (requires JWT token from Firebase Auth)
       # In production, get JWT token from Firebase Auth client
       response = await client.send_message(
           jwt_token=custom_token,  # Or use Firebase ID token
           recipient_id="recipient123",
           message="Hello from Python!"
       )
   ```

**Note:** For production, users should authenticate via Firebase Auth on the frontend and pass the ID token to your backend API.

## Firebase Integration

The agent server can write streaming responses back to Firebase Firestore. This is used in the webhook pattern for mini program integration.

**Setup:**
1. Add Firebase configuration to `.env`:
   ```env
   FIREBASE_PROJECT_ID=your_firebase_project_id
   FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/serviceAccountKey.json
   ```

2. The `firebase_client.py` module handles writing response chunks to Firestore:
   - Chunks are written to: `conversations/{conversation_id}/aiResponses/{message_id}`
   - Mini programs can listen to these updates in real-time

## Development

The backend uses:
- **FastAPI** for the API server
- **LangChain** and **LangGraph** for agentic AI
- **OpenAI** for the language model
- **SSE** for streaming responses
- **Firebase Admin SDK** for writing responses to Firestore
- **Chat21** for chat integration (Python)

