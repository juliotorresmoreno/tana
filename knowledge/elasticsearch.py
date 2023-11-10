
from elasticsearch import Elasticsearch
from knowledge.library import Library
from sentence_transformers import SentenceTransformer
from decouple import config
from elasticsearch.helpers import bulk
import time
from pipe.Pipeline import Response, Arguments
from constants import RESPOND_BASED_ON_CONTEXT

ELASTIC_SEARCH_URL = config("ELASTIC_SEARCH_URL")
ELASTIC_SEARCH_USERNAME = config("ELASTIC_SEARCH_USERNAME")
ELASTIC_SEARCH_PASSWORD = config("ELASTIC_SEARCH_PASSWORD")
ENCODER_MODEL_NAME = config('ENCODER_MODEL_NAME')
DEVICE = config('DEVICE')
ENV = config('ENV')


class ElasticSearchLibrary(Library):
    engine: Elasticsearch
    encoder: SentenceTransformer
    index_name: str

    def __init__(self, index_name: str):
        super().__init__()
        self.index_name = index_name
        self.engine = Elasticsearch(
            ELASTIC_SEARCH_URL,
            basic_auth=(ELASTIC_SEARCH_USERNAME, ELASTIC_SEARCH_PASSWORD),
            verify_certs=ENV == 'production',
            ca_certs="./certs/http_ca.crt",
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

    def invoke(self, args: Arguments) -> Response:
        start_time = time.time()
        vector = self.encoder.encode(args.question)
        knn_search = {
            "knn": {
                "field": "vector",
                "query_vector": vector,
                "k": 3,
                "num_candidates": 20
            },
            "fields": ["content", "command"]
        }

        search_result = self.engine.knn_search(
            index=self.index_name,
            knn=knn_search["knn"],
            fields=knn_search["fields"],
            human=True,
            source=False
        )

        content = ''
        for hit in search_result['hits']['hits']:
            content += '\n' + ''.join(hit['fields']['content'])

        end_time = time.time()
        execution_time = end_time - start_time

        return Response(
            context=content, result=None,
            required_task=RESPOND_BASED_ON_CONTEXT,
            execution_time=execution_time)

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
