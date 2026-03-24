import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

def generate_gemini_answer(prompt: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
    if not API_KEY:
        return "Gemini API Key missing."
    
    # We use requests directly to bypass SDK-specific DNS/connectivity issues
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"Gemini API Error Body: {response.text}")
        response.raise_for_status()
        data = response.json()
        
        # Extract response text
        if "candidates" in data and len(data["candidates"]) > 0:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            return "Gemini returned no candidates."
            
    except Exception as e:
        print("Gemini API error (requests):", e)
        # Try a different model version as fallback if 404
        return "Gemini API error."
