from fastapi import FastAPI, Header, Depends
from fastapi.exceptions import HTTPException

# AUTH_KEY 位置不要在相对引用的代码后面， 会导致循环引用
AUTH_KEY = "489a01a5b35b9c67fc0ebb10a2c7723f65ef30f1204bb199122efd449d897535" # 模拟的认证密钥

from .openai import router as openai_router
from .voyage import router as voyage_router
from .gemini import router as gemini_router
from .ollama import router as ollama_router

def default_authenticate(authorization: str = Header(...)) -> None:
    if authorization != f"Bearer {AUTH_KEY}":
        raise HTTPException(status_code=401, detail="Invalid key")

app = FastAPI()
# 将各部分模拟路由解耦，方便横向扩展
app.include_router(openai_router, prefix="/openai", dependencies=[Depends(default_authenticate)])
app.include_router(voyage_router, prefix="/voyage", dependencies=[Depends(default_authenticate)])
app.include_router(gemini_router, prefix="/gemini") # gemini 每个接口验证逻辑不同，在对应路由页面中单独配置
app.include_router(ollama_router, prefix="/ollama")