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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render."""
    return jsonify({'status': 'healthy', 'service': 'Convo Sol RAG Chatbot'}), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json or {}
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'No question provided.'}), 400

        # Retrieve top relevant chunks for the question
        relevant_chunks = retrieve_top_k(question, top_k=3)
        
        # Extract text from chunks (retrieve_top_k returns list of (score, chunk_dict))
        context = "\n".join([chunk['text'] for score, chunk in relevant_chunks])

        # Build a lead-generation focused prompt
        prompt = (
            f"You are a professional business development chatbot for Convo Sol, an AI SaaS company. "
            f"Your primary goal is to generate leads and convince potential clients to start projects with us. "
            f"Be conversational, brief, and persuasive. Focus on understanding their needs and positioning our services as the solution. "
            f"If they show interest, guide them towards scheduling a meeting to discuss their project in detail.\\n\\n"
            f"Guidelines:\\n"
            f"- Keep responses concise and engaging (2-3 sentences max)\\n"
            f"- Ask follow-up questions to understand their project needs\\n"
            f"- Highlight our expertise and successful track record\\n"
            f"- When appropriate, suggest scheduling a consultation call\\n"
            f"- If clients want to connect with co-founders, provide their contact details from the company information\\n"
            f"- Co-founders: Muneeb Qureshi (muneebq2003@gmail.com, LinkedIn: https://www.linkedin.com/in/muneebqureshi2003/), Muhammad Hadi (muhammadhadiabid@gmail.com, LinkedIn: https://www.linkedin.com/in/muhammad-hadi-abid), Awais Khaleeq (ds.awaisk@gmail.com, LinkedIn: https://www.linkedin.com/in/muhammad-awais-khaleeq)\\n"
            f"- Use a friendly, professional tone like you're chatting with a potential client\\n\\n"
            f"Company Information:\\n{context}\\n\\n"
            f"Client: {question}\\n"
            f"Response:"
        )

        # Call Gemini text-generation API to get the answer
        answer_text = None
        for attempt in range(3):
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                # GenAI SDK returns .text
                answer_text = response.text
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    return jsonify({'error': 'Failed to generate response.'}), 500

        # Fallback if no content was returned
        if not answer_text.strip():
            answer_text = "I'm sorry, I do not have an answer. Please contact support for assistance."
        
        return jsonify({'answer': answer_text})
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

def terminal_chat():
    """Terminal chatbot interface."""
    print("=" * 60)
    print("🤖 RAG Chatbot - Terminal Interface")
    print("=" * 60)
    print("Ask me anything! Type 'quit', 'exit', or 'bye' to stop.")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n💬 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Goodbye!")
                break
                
            if not question:
                continue
                
            print("\n🤖 Bot:", end=" ")
            
            # Use the same logic as the Flask endpoint
            relevant_chunks = retrieve_top_k(question, top_k=3)
            context = "\n".join([chunk['text'] for score, chunk in relevant_chunks])
            
            prompt = (
                f"You are a professional business development chatbot for Convo Sol, an AI SaaS company. "
                f"Your primary goal is to generate leads and convince potential clients to start projects with us. "
                f"Be conversational, brief, and persuasive. Focus on understanding their needs and positioning our services as the solution. "
                f"If they show interest, guide them towards scheduling a meeting to discuss their project in detail.\n\n"
                f"Guidelines:\n"
                f"- Keep responses concise and engaging (2-3 sentences max)\n"
                f"- Ask follow-up questions to understand their project needs\n"
                f"- Highlight our expertise and successful track record\n"
                f"- When appropriate, suggest scheduling a consultation call\n"
                f"- If clients want to connect with co-founders, provide their contact details from the company information\n"
                f"- Co-founders: Muneeb Qureshi (muneebq2003@gmail.com, LinkedIn: https://www.linkedin.com/in/muneebqureshi2003/), Muhammad Hadi (muhammadhadiabid@gmail.com, LinkedIn: https://www.linkedin.com/in/muhammad-hadi-abid), Awais Khaleeq (ds.awaisk@gmail.com, LinkedIn: https://www.linkedin.com/in/muhammad-awais-khaleeq)\n"
                f"- Use a friendly, professional tone like you're chatting with a potential client\n\n"
                f"Company Information:\n{context}\n\n"
                f"Client: {question}\n"
                f"Response:"
            )
            
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                print(response.text)
            except Exception as e:
                print(f"Error: {e}")
                
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'terminal':
        # Run terminal chat
        terminal_chat()
    else:
        # Run Flask server by default
        port = int(os.environ.get('PORT', 8080))
        print(f"🚀 Starting Flask server on http://localhost:{port}")
        print(f"📡 API endpoint: http://localhost:{port}/api/chat")
        print("🔗 Use with ngrok: ngrok http 8080")
        app.run(host='0.0.0.0', port=port, debug=False)
