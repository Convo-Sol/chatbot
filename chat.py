# chat.py
import google.generativeai as genai
import time
from embeddings import get_embedding
from retrieval import InMemoryVectorStore
from config import GEMINI_API_KEY, CHAT_MODEL, TOP_K
from utils import load_db

genai.configure(api_key=GEMINI_API_KEY)

STRICT_SYSTEM_PROMPT = """You are an assistant that MUST ONLY use the provided CONTEXT to answer the user's question.
Do NOT hallucinate, do NOT use outside knowledge. If the answer cannot be found in the context, reply exactly:
"I don't know based on provided documents."

When answering, provide a brief introduction or summary focusing on the most important information only. Do not provide all details at once. If more specifics are needed, suggest the user ask for further details. Be concise. After your answer include a short "Sources:" line listing filename(s) and chunk indices used, for example:
Sources: doc1.txt#2, doc2.txt#0

As a lead generation bot for Convo Sol, always encourage users to contact Convo Sol directly via email (info@convosol.com, support@convosol.com), website (convosol.com), LinkedIn (https://www.linkedin.com/company/convosol/), or contact the co-founders (Muneeb Qureshi: muneebq2003@gmail.com, Muhammad Hadi: muhammadhadiabid@gmail.com, Awais Khaleeq: ds.awaisk@gmail.com) for pricing details, further discussions, or to schedule a meeting.
"""

def build_prompt_from_chunks(question, chunks):
    """
    chunks: list of (score, doc) returned by store.search
    Builds prompt for Gemini API
    We include the top chunks as context.
    """
    context_texts = []
    citations = []
    for i, (score, doc) in enumerate(chunks, start=1):
        # Truncate if very long
        snippet = doc["text"].strip()
        context_texts.append(f"---\nFilename: {doc['filename']}\nChunk: {doc['chunk_index']}\nText:\n{snippet}\n---")
        citations.append(f"{doc['filename']}#{doc['chunk_index']}")

    context_block = "\n\n".join(context_texts)
    prompt = f"""{STRICT_SYSTEM_PROMPT}

Provided context:
{context_block}

User question: {question}

If you can answer, answer and include the Sources line."""
    return prompt, citations

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

def main():
    store = InMemoryVectorStore()
    print("Loaded DB with", len(store.docs), "chunks.")
    print("Enter 'exit' to quit.\n")

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        # Check for greetings and salutations
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings", "howdy"]
        if q.lower() in greetings or any(q.lower().startswith(g) for g in greetings):
            print("\nAssistant:\n")
            print("Hello! How can I help you with ConvoSol today?")
            continue

        q_emb = get_embedding(q, task_type="retrieval_query")
        results = store.search(q_emb)
        # If no results or very low similarity, you may still want to show "I don't know".
        prompt, citations = build_prompt_from_chunks(q, results)
        answer = call_chat_completion(prompt)
        print("\nAssistant:\n")
        print(answer)

if __name__ == "__main__":
    main()
