from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. Initialize the embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to your local Qdrant instance
qdrant = QdrantClient(host="localhost", port=6333)

# 3. Convert your search text into a vector
vec = embedder.encode("test refugee housing support").tolist()
print(f"Vector length: {len(vec)}")

# 4. Search the collection using the updated query_points method
try:
    results = qdrant.query_points(
        collection_name="social_work_reports",
        query=vec,
        limit=3,
        with_payload=True
    ).points
    
    print(f"Search completed. Results count: {len(results)}")
    for r in results:
        print(r.score, r.payload)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")