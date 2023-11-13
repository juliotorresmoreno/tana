from fastapi import APIRouter, WebSocket
from engine.mmlu import engine

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        prompt = await websocket.receive_text()
        if prompt == "" or prompt == None:
            return {"answer": prompt, "response": ""}

        if prompt == "ping":
            return {"answer": prompt, "response": "pong"}

        response, execution_time = engine.invoke(prompt, websocket)

        await websocket.send_json({
            "answer": prompt,
            "response": response,
            "execution_time": execution_time
        })

router_ws = router
