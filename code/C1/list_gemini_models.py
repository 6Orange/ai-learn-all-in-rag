import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# 配置 API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("可用的 Gemini 模型:\n")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"模型名称: {model.name}")
        print(f"  - 支持的方法: {model.supported_generation_methods}")
        print(f"  - 描述: {model.description}")
        print()