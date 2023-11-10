from sentence_transformers import SentenceTransformer
import json
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(
    description="Verifica la existencia de archivos.")

parser.add_argument("collection_name", type=str, help="collection_name")
parser.add_argument("data_path", type=str, help="Ruta al data_path")

args = parser.parse_args()

if not os.path.exists(args.data_path):
    raise Exception("data_path does not exists!")

collection_name = args.collection_name
data_path = args.data_path

print(collection_name)
print(data_path)

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
df = pd.read_json(data_path, lines=True)

vectors = model.encode(
    [str(row.question) + " " + str(row.response) for row in df.itertuples()],
    show_progress_bar=True,
)

QDRANT_URL = os.environ["QDRANT_URL"]
qdrant_client = QdrantClient(QDRANT_URL)

qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

fd = open(data_path)
# payload is now an iterator over startup data
payload = map(json.loads, fd)

"""
qdrant_client.upload_collection(
    collection_name=collection_name,
    vectors=vectors,
    payload=payload,
    ids=None,  # Vector ids will be assigned automatically
    batch_size=256,  # How many vectors will be uploaded in a single request?
)
"""

print(type(payload))

qdrant_client.close()
