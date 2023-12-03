

def parse_token(token: str):
    if not token:
        raise "token is required!"
    
    if token.lower().startswith("bearer"):
        return "token=" + token[7:]
    
    return "token=" + token
