from decouple import config
from app import app

if __name__ == "__main__":
    import uvicorn

    HOST = config("HOST")
    PORT = config("PORT")

    uvicorn.run(app, host=HOST, port=int(PORT))
