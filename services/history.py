import chromadb
from chromadb.api import ClientAPI
from models.user import User
from models.bot import Bot
from langchain.chat_models import ChatOllama
from embeddings.encoder import encoder, HFEncoder
import uuid
from langchain.schema import HumanMessage, SystemMessage, Document
from typing import TypedDict, Literal

chat = ChatOllama()

class Roles(TypedDict):
    human: str
    system: str

roles: Roles = {
    'human': 'human',
    'system': 'system'
}

Format = Literal['messages', 'documents', 'json']

class History:
    conversation: str
    embeddings: HFEncoder
    chromadb: ClientAPI
    user: User
    bot: Bot
    
    def __init__(self, user: User, bot: Bot) -> None:
        self.chromadb = chromadb.HttpClient(host='localhost', port=8000)
        self.user = user
        self.bot = bot
        self.conversation = f"conversation-{user['id']}-{bot['id']}"
        self.embeddings = encoder

    def add_message(self, msg: str, role: str):
        collection = self.chromadb.get_or_create_collection(self.conversation)
        ids = [str(uuid.uuid4())]
        embeddings = [self.embeddings.encode(msg)]
        metadatas = [{ "role": role }]
        documents = [msg]
        collection.add(ids=ids, metadatas=metadatas, documents=documents, embeddings=embeddings)

    def add_title(self, msg: str):
        collection = self.chromadb.get_or_create_collection(self.conversation)
        if collection.count() > 0:
            return
        self.add_system_message(msg=msg)
        
    def add_human_message(self, msg: str):
        self.add_message(msg=msg, role=roles['human'])
        
    def add_system_message(self, msg: str):
        self.add_message(msg=msg, role=roles['system'])

    def get_documents(self, query='', role='', format: Format = 'messages'):
        if format not in ('messages', 'documents', 'json'):
            raise ValueError("Invalid value for 'format'. Must be 'messages', 'documents', or 'json'.")

        collection = self.chromadb.get_or_create_collection(self.conversation)
        if type(query) == str and len(query) > 10:
            if type(role) == str and role != '':
                result = collection.query(query_embeddings=self.embeddings.encode(query), where={
                    "role": { "$eq": role }
                })
            else:
                result = collection.query(query_embeddings=self.embeddings.encode(query))
            documents = result['documents'][0]
            metadatas = result['metadatas'][0]
        else:
            result = collection.get()
            documents = result['documents']
            metadatas = result['metadatas']
            
        response = []
        for document, metadata in zip(documents, metadatas):
            rol = roles['human'] if metadata['role'] == roles['human'] else roles['system']
            if format == 'messages':
                if rol == roles['human']:
                    response.append(HumanMessage(content=document, type=rol))
                else:
                    response.append(SystemMessage(content=document, type=rol))
            elif format == 'documents':
                response.append(Document(page_content=f"{rol}: {document}"))
            else:
                response.append({ 'role': rol, 'content': document })
                
        return response
    
    def get_collection(self):
        collection = self.chromadb.get_or_create_collection(self.conversation)
        return collection.get()
        
    def delete(self):
        try:
            self.chromadb.delete_collection(self.conversation)
        except:
            pass
