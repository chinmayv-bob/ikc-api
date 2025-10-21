from flask import Flask, request, jsonify
from chromadb import PersistentClient
import threading

# Lightweight imports first
import torch
torch.set_num_threads(1)

app = Flask(__name__)

# Connect to your persistent vector store
client = PersistentClient(path="ikc_vector_store")
collection = client.get_collection("ikc_kb")

# Model lazy-load mechanism
model = None
model_lock = threading.Lock()

def get_model():
    """Loads the SentenceTransformer model only when needed."""
    global model
    if model is None:
        with model_lock:
            if model is None:  # double-check inside lock
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return model

@app.route("/ikc_search", methods=["POST"])
def ikc_search():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Query text missing"}), 400

        embedder = get_model()  # load model only now
        query_embedding = embedder.encode(query).tolist()

        results = collection.query(query_embeddings=[query_embedding], n_results=3)

        if results and results["documents"]:
            top_doc = results["documents"][0][0]
            return jsonify({
                "found": True,
                "message": "✅ Related knowledge found in IKC.",
                "snippet": top_doc
            })
        else:
            return jsonify({
                "found": False,
                "message": "❌ No IKC match found.",
                "snippet": None
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "IKC API running", "lazy_model_loaded": model is not None})

if __name__ == "__main__":
    # use 0.0.0.0 for Render/Railway binding
    app.run(host="0.0.0.0", port=10000)
