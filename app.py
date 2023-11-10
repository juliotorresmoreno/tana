from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from decouple import config
from flow import make_pipeline
from pipe.Pipeline import Arguments

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = make_pipeline()


@app.get("/api/ask")
async def ask(prompt: str):
    if prompt == "" or prompt == None:
        return {"answer": prompt, "response": ""}
    
    if prompt == "ping":
        return {"answer": prompt, "response": "pong"}

    response = llm.invoke(Arguments(question=prompt))
    return {"answer": prompt, "response": response.result}

if __name__ == "__main__":
    import uvicorn

    HOST = config("HOST")
    PORT = config("PORT")

    uvicorn.run(app, host=HOST, port=int(PORT))
