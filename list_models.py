import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key present: {api_key is not None}")

try:
    client = genai.Client(api_key=api_key)
    for model in client.models.list():
        print(f"Model Name: {model.name}")
except Exception as e:
    print(f"Error: {e}")
