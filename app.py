from flask import Flask, render_template, request, jsonify
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import random
import itertools
import os
from dotenv import load_dotenv
from functools import lru_cache
import gc

app = Flask(__name__)
load_dotenv()

# -------------------
# Globals (lazy load)
# -------------------
_model = None
_index = None
_data = None

def load_resources():
    global _model, _index, _data
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    if _index is None:
        _index = faiss.read_index("gita_index.faiss")
    if _data is None:
        with open("gita_verses.json", "r", encoding="utf-8") as f:
            _data = json.load(f)

# -------------------
# API Key Rotation
# -------------------
api_keys_str = os.getenv("API_KEYS", "")
API_KEYS = [k.strip() for k in api_keys_str.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("No API keys found. Set API_KEYS in your .env.")
key_cycle = itertools.cycle(API_KEYS)

def get_next_model(model_name: str):
    api_key = next(key_cycle)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

# -------------------
# Query Encoding & Search
# -------------------
@lru_cache(maxsize=500)
def encode_query(query: str):
    load_resources()
    return _model.encode([query], convert_to_numpy=True)

def find_relevant_verses(query, k=3):
    embedding = encode_query(query)
    distances, indices = _index.search(embedding, k)
    return [_data[i] for i in indices[0]]

# -------------------
# Routes
# -------------------
@app.route("/")
def home():
    image_file = f"images/{random.randint(1,15)}.jpg"
    return render_template("index.html", image_file=image_file)

@app.route("/ask-page")
def ask_page():
    image_file = f"images/{random.randint(1,15)}.jpg"
    return render_template("ask.html", image_file=image_file)

@app.route("/tech")
def tech():
    return render_template("tech.html")

@app.route("/health")
def health():
    return "App is running!"

@app.route("/ask", methods=["POST"])
def ask():
    try:
        req = request.json
        name = req.get("name", "Friend")
        age = req.get("age", "unknown")
        query = req.get("query", "")
        language = req.get("language", "English")
        mode = req.get("mode", "normal")

        results = find_relevant_verses(query)

        prompt = f"""
You are Lord Krishna, guiding {name}, age {age}.
Question: {query}
Use these verses: {results}
Instructions:
- Very simple language (English + Hindi + Telugu).
- Quote verses with <b> tags for Sanskrit & translation.
- Space between verses.
- Answer fully in {language}.
- No bullet points. End with: "I, Krishna, am always with you. Be strong."
"""

        llm = get_next_model("gemini-2.5-pro" if mode=="deep" else "gemini-2.5-flash")
        response = llm.generate_content(prompt)

        krishna_response = response.text or ""
        krishna_response = krishna_response.strip("```html").strip("```")

        del response
        gc.collect()

        return jsonify({"response": krishna_response, "verses": results})

    except Exception as e:
        # Log error in Render dashboard
        print("Error in /ask:", e)
        return jsonify({"error": str(e)}), 500

# -------------------
# Run App
# -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


