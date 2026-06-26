from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(host="localhost", port=6333)

vec = embedder.encode("test refugee housing").tolist()
print("Vector length:", len(vec))

try:
    results = qdrant.search(
        collection_name="social_work_reports",
        query_vector=vec,
        limit=3,
        with_payload=True
    )
    print("Results:", len(results))
    for r in results:
        print(r.score, list(r.payload.keys()))
except Exception as e:
    print("ERROR:", type(e).__name__, str(e))
