from flask import Flask, request, jsonify
from flask_cors import CORS
from retrieval import retrieve_top_k
import google.generativeai as genai
from config import GEMINI_API_KEY
import os
import time

app = Flask(__name__)
CORS(app)  # Enable CORS on all routes

# Configure Google GenAI for Gemini
genai.configure(api_key=GEMINI_API_KEY)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided.'}), 400

    # Retrieve top relevant chunks for the question
    relevant_chunks = retrieve_top_k(question, top_k=3)
    # Extract text from chunks (retrieve_top_k returns list of (score, chunk_dict))
    context = "\n".join([chunk['text'] for score, chunk in relevant_chunks])

    # Build a prompt with context and question (with guardrail for accuracy)
    prompt = (
        f"You are a helpful assistant. Use the following context to answer the question "
        f"accurately. If you are unsure, say you do not know.\\n\\n"
        f"Context:\\n{context}\\n\\nQuestion: {question}\\nAnswer:"
    )

    # Call Gemini text-generation API to get the answer
    answer_text = None
    for attempt in range(3):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)
            # GenAI SDK returns .text
            answer_text = response.text
            break
        except Exception:
            if attempt < 2:
                time.sleep(1)
            else:
                return jsonify({'error': 'Failed to generate response.'}), 500

    # Fallback if no content was returned
    if not answer_text.strip():
        answer_text = "I'm sorry, I do not have an answer. Please contact support for assistance."
    return jsonify({'answer': answer_text})

if __name__ == '__main__':
    # Run with one worker (Render sets WEB_CONCURRENCY=1) and no debug mode
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
