# AgenticAIStreaming

A Next.js application that streams AI responses from an agentic AI system to a web app using Firebase, Chat21, LangChain, and LangGraph with OpenAI models.

## Features

- 🤖 **Agentic AI**: Powered by LangChain and LangGraph for intelligent conversation flows
- 🔥 **Firebase Integration**: Real-time database and authentication support
- 💬 **Chat21 Integration**: Chat widget for enhanced messaging experience
- 📡 **Streaming Responses**: Real-time streaming of AI responses using Server-Sent Events
- ⚡ **Next.js 14**: Built with the latest Next.js features including App Router
- 🎨 **Modern UI**: Clean, responsive chat interface with Tailwind CSS

## Tech Stack

- **Frontend**: Next.js 14 (App Router) with TypeScript
- **Backend**: Python FastAPI
- **AI Framework**: LangChain, LangGraph (Python)
- **AI Provider**: OpenAI
- **Backend Services**: Firebase (Firestore, Auth, Storage)
- **Chat Widget**: Chat21
- **Styling**: Tailwind CSS

## Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.9+ and pip
- OpenAI API key
- Firebase project
- Chat21 Cloud Functions deployed to Firebase (optional, for chat integration)

## Setup Instructions

### 1. Install Frontend Dependencies

```bash
cd AgenticAIStreaming
npm install
```

### 2. Setup Python Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Environment Variables

#### Frontend (.env.local)

Create a `.env.local` file in the root directory:

```bash
cp .env.example .env.local
```

Fill in your environment variables:

```env
# Python Backend URL
PYTHON_BACKEND_URL=http://localhost:8000

# Firebase Configuration
NEXT_PUBLIC_FIREBASE_API_KEY=your_firebase_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id

# Chat21 Configuration (Optional)
# Chat21 uses Firebase Cloud Functions - no separate config needed
# Ensure Chat21 Cloud Functions are deployed to your Firebase project
```

### 3. Firebase Setup

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project or use an existing one
3. Enable Firestore Database
4. Enable Authentication (Email/Password provider)
5. Copy your Firebase configuration values to `.env.local`

### 3.1. Chat21 Cloud Functions Setup (Optional)

Chat21 is built on Firebase Cloud Functions. To use Chat21:

1. **Deploy Chat21 Cloud Functions:**
   ```bash
   git clone https://github.com/chat21/chat21-cloud-functions.git
   cd chat21-cloud-functions
   npm install
   firebase login
   firebase use --add  # Select your Firebase project
   firebase deploy
   ```

2. **Enable unauthenticated access** (if needed):
   - Go to Firebase Console → Functions
   - For `/api` and `/supportapi` functions, enable "Allow unauthenticated invocations"
   - Or follow: https://cloud.google.com/functions/docs/securing/managing-access-iam#allowing_unauthenticated_function_invocation

3. **Get Service Account Key** (for backend):
   - Firebase Console → Project Settings → Service Accounts
   - Click "Generate New Private Key"
   - Save JSON file and set `FIREBASE_SERVICE_ACCOUNT_PATH` in backend `.env`

See [Chat21 Cloud Functions README](https://github.com/chat21/chat21-cloud-functions/blob/master/README.md) for detailed setup.

#### Backend (.env)
```env
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Chat21/Firebase Configuration
FIREBASE_PROJECT_ID=your_firebase_project_id
FIREBASE_FUNCTIONS_REGION=us-central1
FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/serviceAccountKey.json
```

### 4. Run the Application

**Terminal 1 - Start Python Backend:**
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Start Next.js Frontend:**
```bash
cd AgenticAIStreaming
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
AgenticAIStreaming/
├── app/
│   ├── api/
│   │   └── chat/
│   │       └── stream/
│   │           └── route.ts      # API route (proxies to Python backend)
│   ├── globals.css                # Global styles
│   ├── layout.tsx                 # Root layout
│   └── page.tsx                   # Home page
├── backend/
│   ├── agent.py                   # LangGraph agent (Python)
│   ├── main.py                    # FastAPI server
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Backend setup instructions
├── components/
│   └── ChatWidget.tsx             # Main chat widget component
├── lib/
│   ├── chat21/
│   │   └── config.ts              # Chat21 configuration
│   └── firebase/
│       ├── config.ts               # Firebase configuration
│       └── messages.ts            # Firebase message utilities
├── .env.example                    # Environment variables template
├── next.config.js                  # Next.js configuration
├── package.json                    # Frontend dependencies
├── tailwind.config.js              # Tailwind CSS configuration
└── tsconfig.json                   # TypeScript configuration
```

## Usage

### Basic Chat Interface

The main chat interface is available at the root route (`/`). Users can:

1. Type messages in the input field
2. Send messages by pressing Enter or clicking the Send button
3. Receive streaming AI responses in real-time
4. View conversation history

### API Endpoint

The streaming API endpoint is available at `/api/chat/stream`:

```typescript
POST /api/chat/stream
Body: {
  message: string,
  conversationHistory?: Array<{role: 'user' | 'assistant', content: string}>
}
```

Response: Server-Sent Events stream with AI response chunks.

## Customization

### Modify the Agent

Edit `backend/agent.py` to customize:

- Agent behavior and prompts
- LangGraph workflow nodes
- Response formatting
- Additional tools or integrations

### Styling

Modify `components/ChatWidget.tsx` and `app/globals.css` to customize the UI appearance.

### Firebase Integration

Extend `lib/firebase/config.ts` to add:

- User authentication
- Message persistence
- Real-time updates
- File storage

## Development

### Build for Production

```bash
npm run build
npm start
```

### Linting

```bash
npm run lint
```

## Notes

- The LangGraph agent uses OpenAI's GPT-4 model by default. You can change this in `lib/agents/langgraph-agent.ts`
- Streaming is handled client-side for better UX. For production, consider implementing server-side streaming with proper error handling
- Chat21 integration requires deploying Chat21 Cloud Functions to your Firebase project. See setup instructions above.
- Firebase and Chat21 integrations are set up but may require additional configuration based on your use case

## License

MIT

