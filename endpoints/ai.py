from fastapi import APIRouter
from engine.mmlu import engine
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import Awaitable, AsyncIterable
import inspect
import asyncio
from embeddings.hf import HFEncoder

encoder = HFEncoder()
router = APIRouter()

class QuestionPayload(BaseModel):
    prompt: str
    index_name: str | None = None

def check_awaitable(obj):
    return inspect.iscoroutinefunction(obj) or inspect.iscoroutine(obj)

async def send_message(payload: QuestionPayload) -> AsyncIterable[str]:
    prompt = payload.prompt
    index_name = payload.index_name
    
    callback = AsyncIteratorCallbackHandler()

    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            if check_awaitable(fn):
                await fn
        except Exception as e:
            print(f"Caught exception: {e}")
        finally:
            event.set()

    task = asyncio.create_task(wrap_done(
        engine.invoke(prompt, index_name, [callback]),
        callback.done),
    )

    async for token in callback.aiter():
        print(token)
        yield token

@router.post("/ai/question")
async def ask(response: StreamingResponse, payload: QuestionPayload):
    prompt = payload.prompt
    if prompt == "" or prompt == None:
        return "Hi, do you need something?"

    if prompt == "ping":
        return "pong"
    
    return StreamingResponse(send_message(payload))

class EmbeddingsPayload(BaseModel):
    prompt: str

@router.post("/ai/embeddings")
async def embeddings(payload: EmbeddingsPayload):
    vector = encoder.encode(payload.prompt)
    return vector

router_ai = router
