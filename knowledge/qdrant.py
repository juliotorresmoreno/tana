from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from decouple import config
from knowledge.library import Library
import time
from pipe.Pipeline import Response, Arguments
from constants import RESPOND_BASED_ON_CONTEXT

DEVICE = config('DEVICE')
ENCODER_MODEL_NAME = config('ENCODER_MODEL_NAME')
QDRANT_URL = config('QDRANT_URL')


class QdrantLibrary(Library):
    engine: QdrantClient
    encoder: SentenceTransformer
    index_name: str
    confidence: float

    def __init__(self, index_name: str, confidence: float = .8):
        super().__init__()

        self.index_name = index_name
        self.confidence = confidence
        self.encoder = SentenceTransformer(ENCODER_MODEL_NAME, device=DEVICE)
        self.engine = QdrantClient(QDRANT_URL)

    def invoke(self, args: Arguments) -> Response:
        if (args.required_task != None and args.required_task != self.task):
            return None, False

        start_time = time.time()
        vector = self.encoder.encode(args.question).tolist()

        search_result = self.engine.search(
            collection_name=self.index_name,
            query_vector=vector,
            query_filter=None,
            score_threshold=.3,
            limit=1
        )
        end_time = time.time()
        execution_time = end_time - start_time

        if len(search_result) > 0 and 'context' in search_result[0].payload:
            current = search_result[0]
            content = current.payload.context['response']

            if (current.score >= self.confidence):
                return Response(
                    context=None, result=content,
                    execution_time=execution_time,
                )

            return Response(
                context=content, result=None,
                execution_time=execution_time,
                required_task=RESPOND_BASED_ON_CONTEXT
            )

        return Response(
            context=None, result=None,
            execution_time=execution_time)
