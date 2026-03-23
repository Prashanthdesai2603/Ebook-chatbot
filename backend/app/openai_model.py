import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

def generate_openai_answer(prompt: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
    if not client:
        return "OpenAI API Key missing or client not initialized."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Defaulting to a strong model
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "OpenAI API error."
