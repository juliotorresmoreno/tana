import json
from embeddings.hf import HFEncoder
from flask import Blueprint, request, Response
from services.auth import Auth
from services.history import History
from services.bots import Bots
from services.llm import LLM
from typing import List
from pydantic import BaseModel, ValidationError, constr
from typing import List
    
auth = Auth()
bots = Bots()
llm = LLM()
router = Blueprint('generate', __name__)
encoder = HFEncoder()

@router.get("/status")
async def status():
    return { "success": "ok" }

class QuestionPayload(BaseModel):
    prompt: constr(min_length=1)
    loaders: List[constr(min_length=1)]

# Función para validar el payload
def validate_question_payload(payload):
    try:
        validated_payload = QuestionPayload(**payload)
        return validated_payload.dict()
    except ValidationError as e:
        print(f"Error validating payload: {e}")
        return None

@router.post("/<int:bot_id>")
def question(bot_id: int):
    headers = request.headers
    authorization = headers.get('authorization')
    validation, session = auth.validate(authorization)
    if not validation:
        return Response(
            json.dumps({ "message": "Authentication is required!" }), 
            content_type="application/json",
            status=401
        )
    
    payload: QuestionPayload = request.json
    if validate_question_payload(payload=payload) == None:
        return Response(
            json.dumps({ "message": "Bad request!" }), 
            content_type="application/json",
            status=400
        )
        
    prompt = payload['prompt']
    loaders = payload["loaders"]
    
    validation, bot = bots.get(authorization, bot_id)
    if not validation:
        return Response(
            json.dumps({ "message": "Bot is required!" }), 
            mimetype="application/json"
        )
    
    chat_history = History(user=session['user'], bot=bot)

    return Response(llm.invoke(
        bot=bot,
        prompt=prompt, 
        loaders=loaders,
        chat_history=chat_history
    ), mimetype="text/event-stream")

router_ai = router
