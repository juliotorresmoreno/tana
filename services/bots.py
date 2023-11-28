import requests
from decouple import config
from utils.authorization import parse_token
from models.bot import Bot

API_URL = config('API_URL')

class Bots:
    def get(self, authorization: str, bot_id: int) -> (bool, Bot):
        if not authorization:
            return False, None
        
        authorization = parse_token(authorization)
        url = f"{API_URL}/mmlu/{str(bot_id)}?{authorization}"
        response = requests.get(url)
        return response.status_code < 300, response.json()
