from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import time
from embeddings import get_embedding
from retrieval import InMemoryVectorStore
from config import GEMINI_API_KEY, CHAT_MODEL, TOP_K
from utils import load_db

genai.configure(api_key=GEMINI_API_KEY)

STRICT_SYSTEM_PROMPT = """You are an assistant that MUST ONLY use the provided CONTEXT to answer the user's question.
Do NOT hallucinate, Do NOT use outside knowledge. If the answer cannot be found in the context, reply exactly:
"I don't know based on provided documents."
Do NOT include any information not found in the context. Rather than making up an answer, politely say to the user that you are an AI Bot for Convo Sol, and you don't know the answer.
As a lead generation bot for Convo Sol, always encourage users to contact Convo Sol directly via email (info@convosol.com, support@convosol.com), website (convosol.com), LinkedIn (https://www.linkedin.com/company/convosol/), or contact the co-founders (Muneeb Qureshi: muneebq2003@gmail.com, Muhammad Hadi: muhammadhadiabid@gmail.com, Awais Khaleeq: ds.awaisk@gmail.com) for pricing details, further discussions, or to schedule a meeting.
"""

def build_prompt_from_chunks(question, chunks):
    """
    chunks: list of (score, doc) returned by store.search
    Builds prompt for Gemini API
    We include the top chunks as context.
    """
    context_texts = []
    for i, (score, doc) in enumerate(chunks, start=1):
        # Truncate if very long
        snippet = doc["text"].strip()
        context_texts.append(f"---\n{snippet}\n---")

    context_block = "\n\n".join(context_texts)
    prompt = f"""{STRICT_SYSTEM_PROMPT}

Provided context:
{context_block}

User question: {question}

If you can answer, answer."""
    return prompt

def call_chat_completion(prompt, model=CHAT_MODEL, max_tokens=512):
    for attempt in range(3):
        try:
            # SDK expects model name without 'models/' prefix
            model_name = model.replace('models/', '') if model.startswith('models/') else model
            gen_model = genai.GenerativeModel(model_name)
            response = gen_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.0
                )
            )
            # Handle response text extraction
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("Unexpected response format")
        except Exception as e:
            wait = 2 ** attempt
            print(f"Chat error: {e}; retrying in {wait}s")
            if attempt == 0:  # On first attempt, try to list available models
                try:
                    print("Available models:", [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods])
                except:
                    pass
            time.sleep(wait)
    raise RuntimeError("Failed chat request")

def get_answer(question):
    store = InMemoryVectorStore()
    # Check for greetings and salutations
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings", "howdy"]
    if question.lower() in greetings or any(question.lower().startswith(g) for g in greetings):
        return "Hello! How can I help you with ConvoSol today?"

    q_emb = get_embedding(question, task_type="retrieval_query")
    results = store.search(q_emb)
    # If no results or very low similarity, you may still want to show "I don't know".
    prompt = build_prompt_from_chunks(question, results)
    answer = call_chat_completion(prompt)
    return answer

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True, silent=True)
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing question'}), 400
    question = data['question'].strip()
    answer = get_answer(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
