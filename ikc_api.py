from flask import Flask, request, jsonify
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# 🚀 Initialize Flask app
app = Flask(__name__)

# 🧠 Load your local Chroma Vector DB
client = PersistentClient(path="C:/Users/chinm/desktop/ikc/ikc_vector_store")
collection = client.get_collection("ikc_kb")

# 🔧 Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route('/ikc_search', methods=['POST'])
def ikc_search():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Embed the query and search top 3 results
        emb = model.encode(query).tolist()
        results = collection.query(query_embeddings=[emb], n_results=3)

        top_docs = results['documents'][0] if results['documents'] else []
        scores = results['distances'][0] if results['distances'] else []

        if not top_docs:
            return jsonify({"found": False, "message": "❌ No relevant info found in IKC"})

        top_match = top_docs[0]
        score = round(scores[0], 4)

        return jsonify({
            "found": True,
            "message": f"✅ Related info found in IKC (score: {score})",
            "top_snippet": top_match[:400],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
