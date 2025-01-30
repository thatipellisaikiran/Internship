import sqlite3
import faiss
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")
d = 384 
index = faiss.IndexFlatL2(d)
with open("The ICC Champions Trophy was an int.txt", "r") as file:
    file_content = file.read()
corpus =file_content.split("\n")
corpus_embeddings = model.encode(corpus)
index.add(np.array(corpus_embeddings, dtype=np.float32))
conn = sqlite3.connect("rag_chatbot.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    role TEXT CHECK(role IN ('user', 'system')),
    content TEXT
)
''')
conn.commit()

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query")
    cursor.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", ("user", user_query))
    conn.commit()
    query_embedding = model.encode([user_query])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), k=1)
    retrieved_text = corpus[indices[0][0]]
    response = f"Here’s relevant info: {retrieved_text}"
    cursor.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", ("system", response))
    conn.commit()
    return jsonify({"response": response})

@app.route("/history", methods=["GET"])
def history():
    cursor.execute("SELECT timestamp, role, content FROM chat_history ORDER BY timestamp ASC")
    history_data = cursor.fetchall()
    return jsonify(history_data)

if __name__ == "__main__":
    app.run(debug=True)
