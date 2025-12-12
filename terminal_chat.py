#!/usr/bin/env python3
"""
Terminal-based chatbot using the same RAG system as the Flask app.
Run with: python terminal_chat.py
"""

from retrieval import retrieve_top_k
import google.generativeai as genai
from config import GEMINI_API_KEY
import time

# Configure Google GenAI for Gemini
genai.configure(api_key=GEMINI_API_KEY)

def get_answer(question):
    """Get answer for a question using RAG system."""
    try:
        print("🔍 Searching for relevant information...")
        
        # Retrieve top relevant chunks for the question
        relevant_chunks = retrieve_top_k(question, top_k=3)
        print(f"📚 Found {len(relevant_chunks)} relevant chunks")
        
        # Extract text from chunks
        context = "\n".join([chunk['text'] for score, chunk in relevant_chunks])

        # Build a prompt with context and question
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

        print("🤖 Generating response...")
        
        # Call Gemini API to get the answer
        for attempt in range(3):
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                answer_text = response.text
                
                if answer_text and answer_text.strip():
                    return answer_text
                else:
                    return "I'm sorry, I do not have an answer. Please contact support for assistance."
                    
            except Exception as e:
                print(f"⚠️  API error (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(1)
                else:
                    return "Failed to generate response. Please try again later."
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        return "An error occurred while processing your question."

def main():
    """Main chatbot loop."""
    print("=" * 60)
    print("🤖 RAG Chatbot - Terminal Interface")
    print("=" * 60)
    print("Ask me anything! Type 'quit', 'exit', or 'bye' to stop.")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            question = input("\n💬 You: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Goodbye! Thanks for chatting!")
                break
                
            # Skip empty questions
            if not question:
                print("❓ Please ask a question.")
                continue
                
            # Get and display answer
            print("\n🤖 Bot:", end=" ")
            answer = get_answer(question)
            print(answer)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again.")

if __name__ == '__main__':
    main()