OPENAI_ENDPOINT = "http://localhost:9000/openai" # 模拟的 openai 接口地址
VOYAGE_ENDPOINT = "http://localhost:9000/voyage" # 模拟的 voyage 接口地址
GEMINI_ENDPOINT = "http://localhost:9000/gemini" # 模拟的 gemini 接口地址
OLLAMA_ENDPOINT = "http://localhost:9000/ollama" # 模拟的 ollama 接口地址
REFERENCE_VECTOR: list[float] = [round(i* 0.05, 2) for i in range(20)] # 用于给模拟 api 返回向量和 pytest assert 验证使用

from .app import app, AUTH_KEY

__all__ = [
    "app",
    "AUTH_KEY", 
    "OPENAI_ENDPOINT", 
    "VOYAGE_ENDPOINT",
    "GEMINI_ENDPOINT",
    "OLLAMA_ENDPOINT",
    "REFERENCE_VECTOR"
]