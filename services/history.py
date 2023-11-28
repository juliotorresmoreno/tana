import chromadb
from models.user import User
from models.bot import Bot
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOllama()

class History:
    def __init__(self, user: User, bot: Bot) -> None:
        self.chromadb = chromadb.HttpClient(host='localhost', port=8000)
        self.user = user
        self.bot = bot

    def get():
        pass