# ==============================================
# IKC Vector Database Builder (Local Embeddings)
# ==============================================
import sys, os
sys.path.append(r"C:\Users\chinm\AppData\Roaming\Python\Python311\site-packages")


import chromadb
from sentence_transformers import SentenceTransformer
import re
import os

# --- Config ---
IKC_FILE = "ikc.txt"
DB_PATH = "./ikc_vector_store"
COLLECTION_NAME = "ikc_kb"

# --- Setup ---
print("🚀 Initializing Chroma vector database (local mode)...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

print("🔧 Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Step 1: Read and clean IKC file ---
if not os.path.exists(IKC_FILE):
    raise FileNotFoundError(f"❌ File '{IKC_FILE}' not found in current folder!")

with open(IKC_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# --- Step 2: Split text into chunks ---
# Split by bullets, dashes, or paragraph breaks
chunks = re.split(r"[\n•\-–]+", text)
chunks = [c.strip() for c in chunks if len(c.strip()) > 30]

print(f"📄 Extracted {len(chunks)} meaningful chunks from IKC.")

# --- Step 3: Embed and store in Chroma ---
for i, chunk in enumerate(chunks):
    emb = model.encode(chunk).tolist()
    collection.add(
        ids=[f"ikc_{i}"],
        embeddings=[emb],
        documents=[chunk],
        metadatas=[{"source": "IKC"}]
    )
    if (i + 1) % 10 == 0:
        print(f"✅ Added {i + 1}/{len(chunks)} chunks")

print("🎯 IKC Vector Database created successfully!")
print(f"💾 Stored in: {os.path.abspath(DB_PATH)}")

# --- Step 4: Optional quick query ---
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # local model, already cached

while True:
    query = input("\n🔍 Enter a query (or press Enter to exit): ").strip()
    if not query:
        print("👋 Done!")
        break
    q_emb = model.encode(query).tolist()
    result = collection.query(query_embeddings=[q_emb], n_results=3)
    print("\nTop matches:")
    for doc, dist in zip(result["documents"][0], result["distances"][0]):
        print(f"• Score {dist:.4f}\n  {doc[:300]}\n")
