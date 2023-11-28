import json
from embeddings.hf import HFEncoder
from flask import Blueprint, request, Response
from typing import TypedDict
from services.auth import Auth
from services.history import History
from services.bots import Bots
from services.llm import LLM
from models.session import Session
from models.bot import Bot

auth = Auth()
bots = Bots()
llm = LLM()
router = Blueprint('ai', __name__)
encoder = HFEncoder()
history = History()

@router.get("/status")
async def status():
    return { "success": "ok" }

@router.delete("/conversation/<int:bot_id>")
async def delete(bot_id: int):
    headers = request.headers
    authorization = headers.get('authorization')
    validation, session = await auth.validate(authorization)
    if not validation:
        return Response(
            json.dumps({ "message": "Authentication is required!" }), 
            mimetype="text/event-stream"
        )
    history.delete(session['user'], bot_id)
    return Response('', 204)

@router.get("/conversation/<int:bot_id>")
def conversation(bot_id: int):
    headers = request.headers
    authorization = headers.get('authorization')
    validation, session = auth.validate(authorization)
    if not validation:
        return Response(
            json.dumps({ "message": "Authentication is required!" }), 
            mimetype="text/event-stream"
        )
    return history.get(session['user'], bot_id)

@router.post("/question/<int:bot_id>")
def question(bot_id: int):
    headers = request.headers
    authorization = headers.get('authorization')
    validation, session: Session = auth.validate(authorization)
    if not validation:
        return Response(
            response=json.dumps({ "message": "Authentication is required!" }), 
            mimetype="application/json", status=401
        )
    
    prompt: str = request.json['prompt']
    index_name = None
    if "index_name" in request.json:
        index_name = request.json['index_name']

    if prompt == "" or prompt == None:
        return "Hi, do you need something?"

    if prompt == "ping":
        return "pong"
    
    validation, bot: Bot = bots.get(authorization=authorization, bot_id=bot_id)
    if not validation:
        return Response(
            json.dumps({ "message": "Authentication is requires!" }), 
            mimetype="application/json",
            status=400
        )
    
    chat_history = History(user=session['user'], bot=bot)

    return Response(llm.invoke(
        prompt=prompt, 
        index_name=index_name,
        chat_history=chat_history
    ), mimetype="text/event-stream")

class EmbeddingsPayload(TypedDict):
    prompt: str

@router.post("/embeddings")
async def embeddings():
    payload: EmbeddingsPayload = request.json
    
    if "prompt" not in payload:
        return Response(json.dumps({ "message": "prompt is required!" }), 400)
    
    return encoder.encode(payload["prompt"])

router_ai = router
