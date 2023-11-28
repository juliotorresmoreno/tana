from typing import TypedDict
from models.user import User

class Session(TypedDict):
    token: str
    user: User
    