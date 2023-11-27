from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from endpoints.home import router_home
from endpoints.ws import router_ws
from endpoints.ai import router_ai

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router_home)
app.include_router(router_ws)
app.include_router(router_ai)
