from decouple import config
from langchain.embeddings import GPT4AllEmbeddings

ENCODER_MODEL_NAME = config('ENCODER_MODEL_NAME')
DEVICE = config('DEVICE')

class HFEncoder:
    encoder: GPT4AllEmbeddings

    def __init__(self):
        super().__init__()
        self.encoder = GPT4AllEmbeddings()
        
    def encode(self, prompt: str):
        return self.encoder.embed_query(prompt)