import google.generativeai as genai
genai.configure(api_key="AIzaSyD82ZzFYCyU3GMdV4J6DKqz7vU19bBOL3o")
models = genai.list_models()
for m in models:
    print(m.name, m.supported_generation_methods)
