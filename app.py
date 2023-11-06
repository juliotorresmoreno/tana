from neuralsearcher import NeuralSearcher
from fastapi import FastAPI
from dotenv import load_dotenv
import os
load_dotenv()

app = FastAPI()

neural_searcher = NeuralSearcher(collection_name='root')


@app.get("/api/ask")
async def search(q: str):
    if q == "" or q == None:
        return {"result": []}
    result = neural_searcher.search(text=q, limit=1)

    if result == None:
        return {"result": []}
    
    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    HOST = os.environ["HOST"]
    PORT = os.environ["PORT"]

    uvicorn.run(app, host=HOST, port=int(PORT))
