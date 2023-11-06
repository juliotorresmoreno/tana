from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os


class NeuralSearcher:
    def __init__(self, collection_name):
        DEVICE = os.environ["DEVICE"]
        MODEL_NAME = os.environ["MODEL_NAME"]
        QDRANT_URL = os.environ["QDRANT_URL"]

        self.collection_name = collection_name
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        self.qdrant_client = QdrantClient(QDRANT_URL)

    def search(self, text: str, limit: int = 5):
        vector = self.model.encode(text).tolist()

        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            score_threshold=.3,
            limit=limit
        )
        payloads = [hit.payload for hit in search_result]

        return payloads

    def query(self, key: str, text: str):
        vector = self.model.encode(text).tolist()

        filter_key = Filter(**{
            "must": [{
                "key": key,
                "match": {
                    "value": text
                }
            }]
        })

        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=filter_key,
            limit=5
        )

        payloads = [hit.payload for hit in search_result]
        return payloads
