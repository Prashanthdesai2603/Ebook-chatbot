import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
data = {
    "contents": [{"parts": [{"text": "Say hello"}]}]
}

try:
    response = requests.post(url, json=data)
    print("Status:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Failed:", e)
