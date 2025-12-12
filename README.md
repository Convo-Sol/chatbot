# Convo Sol RAG Chatbot

AI-powered lead generation chatbot for Convo Sol using RAG (Retrieval-Augmented Generation) with ChromaDB and Google Gemini.

## Features

- Professional lead generation chatbot
- RAG-based responses using company knowledge base
- Co-founder contact information sharing
- Flask API for frontend integration
- Terminal chat interface for testing

## Environment Variables

Set these in your Render environment:

```
GEMINI_API_KEY=your_gemini_api_key_here
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHAT_MODEL=gemini-2.5-flash
TOP_K=4
CHUNK_SIZE=800
CHUNK_OVERLAP=200
```

## API Endpoints

- `GET /health` - Health check
- `POST /api/chat` - Chat endpoint

### Chat API Usage

```bash
curl -X POST https://your-app.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What services do you offer?"}'
```

## Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Set up `.env` file with your API keys
3. Run Flask server: `python app.py`
4. Run terminal chat: `python app.py terminal`

## Deployment

This app is configured for Render deployment with:
- Procfile for Gunicorn configuration
- Health check endpoint
- Production-ready settings