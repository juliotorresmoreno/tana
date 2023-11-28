

def parse_token(token: str):
    if not token:
        raise "token is required!"
    
    if token.lower().startswith("bearer"):
        return "authorization=" + token[7:]
    
    return "authorization=" + token
