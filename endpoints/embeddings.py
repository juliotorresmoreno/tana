import json
from flask import Blueprint, request, Response
from services.auth import Auth
from embeddings.encoder import encoder
from pydantic import BaseModel, ValidationError, constr

auth = Auth()
router = Blueprint('embeddings', __name__)

@router.get("status")
async def status():
    return { "success": "ok" }

class EmbeddingsPayload(BaseModel):
    prompt: constr(min_length=1)

def validate_embeddings_payload(payload):
    try:
        validated_payload = EmbeddingsPayload(**payload)
        return validated_payload.dict()
    except ValidationError as e:
        print(f"Error validating payload: {e}")
        return None

@router.post("")
async def embeddings():
    payload: EmbeddingsPayload = request.json
    
    payload: EmbeddingsPayload = request.json
    if validate_embeddings_payload(payload=payload) == None:
        return Response(
            json.dumps({ "message": "Bad request!" }), 
            content_type="application/json",
            status=400
        )
    
    return encoder.encode(payload["prompt"])

router_embeddings = router
