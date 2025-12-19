# Architecture: Mini Program → Firebase → Chat21 → Agent Server

## Overview

This architecture supports mini program clients (WeChat, Alipay, etc.) through Firebase and Chat21 integration with an AI agent backend.

## System Components

### 1. Mini Program (Client)
- Sends messages to Firebase Firestore/Realtime Database
- Listens to Firebase updates for AI responses
- Displays streaming responses in real-time

### 2. Firebase (Message Broker)
- **Firestore/Realtime Database**: Stores messages and conversation state
- **Cloud Functions**: Chat21 functions that process incoming messages
- **Real-time Listeners**: Notifies mini program of updates

### 3. Chat21 Server (Firebase Cloud Functions)
- Receives messages from Firebase
- Manages chat conversations
- Calls Agent Server webhook when AI processing is needed
- Handles message routing and delivery

### 4. Agent Server (Python FastAPI)
- Receives webhook calls from Chat21
- Processes messages through LangGraph agent
- Streams responses back to Firebase
- Updates conversation state in Firestore

## Message Flow

```
┌─────────────┐
│ Mini Program│
└──────┬──────┘
       │ 1. Send message
       ▼
┌─────────────────┐
│   Firebase      │
│  (Firestore)    │
└──────┬──────────┘
       │ 2. Message stored
       │    Triggers Cloud Function
       ▼
┌─────────────────┐
│  Chat21 Server  │
│ (Cloud Function)│
└──────┬──────────┘
       │ 3. Detect AI message
       │    Call Agent Server
       ▼
┌─────────────────┐
│  Agent Server   │
│  (Python API)   │
└──────┬──────────┘
       │ 4. Process with LangGraph
       │    Stream response chunks
       ▼
┌─────────────────┐
│   Firebase      │
│  (Firestore)    │
└──────┬──────────┘
       │ 5. Update conversation
       │    Real-time update
       ▼
┌─────────────┐
│ Mini Program│
│ (receives)  │
└─────────────┘
```

## Detailed Flow

### Step 1: Mini Program → Firebase
- User sends message in mini program
- Message saved to Firestore collection: `conversations/{conversationId}/messages`
- Message structure:
  ```json
  {
    "id": "msg123",
    "senderId": "user123",
    "text": "Hello, how are you?",
    "timestamp": "2024-01-01T00:00:00Z",
    "type": "text",
    "needsAI": true
  }
  ```

### Step 2: Firebase → Chat21
- Firebase Cloud Function (Chat21) triggered on new message
- Function checks if message needs AI processing (`needsAI: true`)
- Function validates message and conversation

### Step 3: Chat21 → Agent Server
- Chat21 Cloud Function calls Agent Server webhook:
  ```
  POST https://agent-server.com/api/webhook/chat21
  {
    "conversationId": "conv123",
    "messageId": "msg123",
    "userId": "user123",
    "message": "Hello, how are you?",
    "conversationHistory": [...]
  }
  ```

### Step 4: Agent Server Processing
- Agent Server receives webhook
- Processes message through LangGraph agent
- Streams response chunks back to Firebase:
  ```json
  {
    "conversationId": "conv123",
    "messageId": "msg123",
    "chunk": "Hello!",
    "isComplete": false
  }
  ```

### Step 5: Firebase → Mini Program
- Agent Server writes chunks to Firestore: `conversations/{conversationId}/aiResponses/{messageId}`
- Mini program listens to Firestore updates
- UI updates in real-time as chunks arrive
- Final chunk marked with `isComplete: true`

## Implementation Details

### Agent Server Endpoints

#### POST /api/webhook/chat21
Receives webhook from Chat21 Cloud Function.

**Request:**
```json
{
  "conversationId": "string",
  "messageId": "string",
  "userId": "string",
  "message": "string",
  "conversationHistory": [
    {"role": "user|assistant", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "status": "accepted",
  "conversationId": "string",
  "messageId": "string"
}
```

The agent then streams response chunks to Firebase asynchronously.

### Firebase Structure

```
conversations/
  {conversationId}/
    messages/
      {messageId}/
        - text: string
        - senderId: string
        - timestamp: timestamp
        - type: string
        - needsAI: boolean
    aiResponses/
      {messageId}/
        - chunks: array
        - currentText: string
        - isComplete: boolean
        - updatedAt: timestamp
```

### Chat21 Cloud Function

The Chat21 Cloud Function needs to:
1. Listen to new messages in Firestore
2. Check if `needsAI: true`
3. Call Agent Server webhook
4. Handle errors and retries

## Environment Variables

### Agent Server (.env)
```env
OPENAI_API_KEY=your_key
FIREBASE_PROJECT_ID=your_project
FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/serviceAccount.json
AGENT_SERVER_URL=https://your-agent-server.com
```

### Chat21 Cloud Functions
```env
AGENT_SERVER_WEBHOOK_URL=https://your-agent-server.com/api/webhook/chat21
FIREBASE_PROJECT_ID=your_project
```

## Benefits

1. **Scalability**: Firebase handles real-time updates efficiently
2. **Reliability**: Chat21 manages message delivery and retries
3. **Separation of Concerns**: Agent server focuses on AI processing
4. **Mini Program Support**: Native Firebase SDK support for mini programs
5. **Real-time Updates**: Firestore listeners provide instant updates

## Next Steps

1. Implement webhook endpoint in Agent Server
2. Add Firebase Admin SDK integration for writing responses
3. Create Chat21 Cloud Function that calls Agent Server
4. Update Chat21 client to handle agent callbacks
5. Test end-to-end flow

