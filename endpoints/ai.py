from fastapi import APIRouter
from engine.mmlu import engine
from pydantic import BaseModel

router = APIRouter()


class QuestionPayload(BaseModel):
    prompt: str
    index_name: str | None = None

@router.post("/ai/question")
async def ask(payload: QuestionPayload):
    prompt = payload.prompt
    index_name = payload.index_name
    if prompt == "" or prompt == None:
        return {"answer": prompt, "response": ""}

    if prompt == "ping":
        return {"answer": prompt, "response": "pong"}

    response, execution_time = engine.invoke(prompt, index_name=index_name)
    return {"answer": prompt, "response": response, "execution_time": execution_time}

router_ai = router
