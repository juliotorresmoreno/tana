
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from decouple import config
from elasticsearch.helpers import bulk
from langchain.docstore.document import Document

ELASTIC_SEARCH_URL = config("ELASTIC_SEARCH_URL")
ELASTIC_SEARCH_USERNAME = config("ELASTIC_SEARCH_USERNAME")
ELASTIC_SEARCH_PASSWORD = config("ELASTIC_SEARCH_PASSWORD")
ENCODER_MODEL_NAME = config('ENCODER_MODEL_NAME')
ELASTIC_SEARCH_CERT = config('ELASTIC_SEARCH_CERT')
DEVICE = config('DEVICE')
ENV = config('ENV')


class ElasticSearchLoader:
    engine: Elasticsearch
    encoder: SentenceTransformer

    def __init__(self):
        super().__init__()
        self.engine = Elasticsearch(
            ELASTIC_SEARCH_URL,
            basic_auth=(ELASTIC_SEARCH_USERNAME, ELASTIC_SEARCH_PASSWORD),
            verify_certs=True,
            ca_certs=ELASTIC_SEARCH_CERT,
        )
        self.encoder = SentenceTransformer(
            ENCODER_MODEL_NAME, device=DEVICE)

    def create_index(self):
        index_settings = {
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text"
                    },
                    "url": {
                        "type": "text"
                    },
                    "vector": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "content": {
                        "type": "text"
                    },
                    "command": {
                        "type": "text"
                    },
                    "keyword": {
                        "type": "keyword"
                    }
                }
            }
        }
        self.engine.indices.create(
            index=self.index_name,
            body=index_settings
        )

    def search(self, index_name: str, question: str):
        vector = self.encoder.encode(question)
        knn_search = {
            "knn": {
                "field": "vector",
                "query_vector": vector,
                "k": 10,
                "num_candidates": 20
            },
            "fields": ["content", "command", "title", 'url']
        }

        search_result = self.engine.knn_search(
            index=index_name,
            knn=knn_search["knn"],
            fields=knn_search["fields"],
            human=True,
            source=False
        )

        documents = []
        for hit in search_result['hits']['hits']:
            content = hit['fields']['content'][0][:10]
            title = hit['fields']['title'][0]
            url = hit['fields']['url'][0]

            documents.append(Document(
                page_content=content,
                metadata={"source": url, "title": title}
            ))

        return documents

    def create_bulk_actions(self, data_array):
        actions = []
        for data in data_array:
            action = {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": data["id"],
                "_source": {
                    "title": data["title"],
                    "vector": self.encoder.encode(data["content"]),
                    "content": data["content"],
                    "keyword": data["keyword"],
                    "command": data["command"],
                    "url": data["url"],
                }
            }
            actions.append(action)
        return actions

    def add(self, data_array: list):
        bulk_actions = self.create_bulk_actions(data_array)
        success, failed = bulk(self.engine, bulk_actions, refresh=True)

        return success, failed
