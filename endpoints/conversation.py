import json
from embeddings.hf import HFEncoder
from flask import Blueprint, request, Response
from services.auth import Auth
from services.history import History
from services.bots import Bots
from services.llm import LLM

auth = Auth()
bots = Bots()
llm = LLM()
router = Blueprint('conversation', __name__)
encoder = HFEncoder()

@router.get("status")
def status():
    return { "success": "ok" }

@router.delete("<int:bot_id>")
def delete(bot_id: int):
    headers = request.headers
    authorization = headers.get('authorization')
    validation, session = auth.validate(authorization)
    if not validation:
        return Response(
            json.dumps({ "message": "Authentication is required!" }), 
            mimetype="application/json"
        )
    
    validation, bot = bots.get(authorization, bot_id)
    if not validation:
        return Response(
            json.dumps({ "message": "Bot is required!" }), 
            mimetype="application/json"
        )
    History(user=session['user'], bot=bot).delete()
    return Response('', 204)

@router.get("/<int:bot_id>")
def conversation(bot_id: int):
    headers = request.headers
    authorization = headers.get('authorization')
    validation, session = auth.validate(authorization)
    if not validation:
        return Response(
            json.dumps({ "message": "Authentication is required!" }), 
            mimetype="application/json"
        )
    
    validation, bot = bots.get(authorization, bot_id)
    if not validation:
        return Response(
            json.dumps({ "message": "Bot is required!" }), 
            mimetype="application/json"
        )
    
    role = request.args.get('role')
    query = request.args.get('q')

    chat_history = History(user=session['user'], bot=bot).get_documents(query=query, role=role, format='json')
    return Response(json.dumps(chat_history), 200, {"content-type": "application/json"})


router_conversation = router
