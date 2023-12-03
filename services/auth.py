import requests
from decouple import config
from utils.authorization import parse_token
from models.session import Session
API_URL = config('API_URL')

class Auth:
    def validate(self, authorization: str) -> (bool, Session):
        if not authorization:
            return False, None
        
        authorization = parse_token(authorization)
        url = f"{API_URL}/api/auth/session?{authorization}"
        response = requests.get(url)
        
        if response.status_code < 300:
            return True, response.json()

        return False, None
