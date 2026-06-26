import zipfile
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
ZIP_PATH       = r"C:\Users\Leon\Downloads\Sample Reports.zip"
EXTRACT_FOLDER = r"C:\Users\Leon\Downloads\Sample Reports"
COLLECTION     = "social_work_reports"
CHUNK_SIZE     = 500   # characters per chunk

# ── STEP 1: UNZIP ──────────────────────────────────────────────────────────────
print("📦 Unzipping reports...")
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_FOLDER)
print(f"✅ Extracted to: {EXTRACT_FOLDER}")

# ── STEP 2: FIND ALL PDFs ─────────────────────────────────────────────────────
pdf_files = []
for root, dirs, files in os.walk(EXTRACT_FOLDER):
    for f in files:
        if f.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(root, f))

print(f"\n📄 Found {len(pdf_files)} PDF files:")
for p in pdf_files:
    print(f"   {p}")

if not pdf_files:
    print("❌ No PDFs found. Check your zip file contents.")
    exit()

# ── STEP 3: EXTRACT TEXT AND CHUNK ────────────────────────────────────────────
print("\n📖 Reading and chunking PDFs...")

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    current = []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            # overlap: keep last 20% of words
            overlap = max(1, len(current) // 5)
            current = current[-overlap:]
            current_len = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return chunks

all_chunks = []  # list of dicts: {text, source}

for pdf_path in pdf_files:
    filename = os.path.basename(pdf_path)
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        if not full_text.strip():
            print(f"   ⚠️  {filename} — no text extracted (may be scanned image)")
            continue

        chunks = chunk_text(full_text)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": filename})

        print(f"   ✅ {filename} → {len(chunks)} chunks")
    except Exception as e:
        print(f"   ❌ {filename} — error: {e}")

print(f"\n📊 Total chunks to embed: {len(all_chunks)}")

# ── STEP 4: EMBED ─────────────────────────────────────────────────────────────
print("\n🔢 Loading embedding model (this may take a minute)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🔢 Embedding all chunks...")
texts = [c["text"] for c in all_chunks]
embeddings = embedder.encode(texts, show_progress_bar=True)
print("✅ Embeddings complete")

# ── STEP 5: LOAD INTO QDRANT ──────────────────────────────────────────────────
print("\n🗄️  Connecting to Qdrant...")
qdrant = QdrantClient(host="localhost", port=6333)

# Delete collection if it already exists (fresh start)
existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION in existing:
    print(f"   ⚠️  Collection '{COLLECTION}' already exists — deleting for fresh load...")
    qdrant.delete_collection(COLLECTION)

# Create collection
qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
print(f"   ✅ Collection '{COLLECTION}' created")

# Insert points
points = []
for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={"text": chunk["text"], "source": chunk["source"]}
    ))

# Upload in batches of 100
batch_size = 100
for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    qdrant.upsert(collection_name=COLLECTION, points=batch)
    print(f"   Uploaded {min(i+batch_size, len(points))}/{len(points)} chunks...")

print(f"\n🎉 Done! {len(points)} chunks from {len(pdf_files)} reports loaded into Qdrant.")
print(f"   Collection: {COLLECTION}")
print(f"   You can verify at: http://localhost:6333/collections/{COLLECTION}")
