from sentence_transformers import SentenceTransformer
import time
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from datasets import load_dataset
from decouple import config

collection_name = "wiki"

QDRANT_URL = config("QDRANT_URL")

qdrant_client = QdrantClient(QDRANT_URL)
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
dataset = load_dataset("graelo/wikipedia", "20230601.en")
# dataset = pd.read_json('data/wiki.json')

batch_size = 256 * 10

df = dataset['train']


# Itera a través de los registros de 100 en 100
#for i in range(0, len(df), batch_size):
    
records = df # df[i:i + batch_size]

payload = [{
    "id": int(records['id'][record]),
    "title": records['title'][record],
    "text": records['text'][record],
    "url": records['url'][record]
} for record in range(len(records['id']))]

vectors = model.encode(
    records['text'],
    show_progress_bar=True,
)

qdrant_client.upload_collection(
    collection_name=collection_name,
    vectors=vectors,
    payload=payload,
    ids=None,
    batch_size=256,
    wait=True
)

qdrant_client.close()

time.sleep(10)
