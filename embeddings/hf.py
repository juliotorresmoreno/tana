from sentence_transformers import SentenceTransformer
from decouple import config

ENCODER_MODEL_NAME = config('ENCODER_MODEL_NAME')
DEVICE = config('DEVICE')

class HFEncoder:
    encoder: SentenceTransformer

    def __init__(self):
        super().__init__()
        self.encoder = SentenceTransformer(ENCODER_MODEL_NAME, device=DEVICE)
        
    def encode(self, prompt: str):
        return self.encoder.encode(prompt).tolist()
