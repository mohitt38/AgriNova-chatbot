import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(GOOGLE_API_KEY)
models = genai.list_models()
for m in models:
    print(m.name, m.supported_generation_methods)
