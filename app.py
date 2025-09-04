from flask import Flask, render_template, request, jsonify
import faiss
import json
import numpy as np
import google.generativeai as genai
import random
import itertools
import os
from dotenv import load_dotenv
import gc

# -------------------
# Setup Flask + dotenv
# -------------------
app = Flask(__name__)
load_dotenv()

# -------------------
# Load API Keys
# -------------------
api_keys_str = os.getenv("API_KEYS", "")
API_KEYS = [k.strip() for k in api_keys_str.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("No API keys found. Please set API_KEYS in your environment.")
key_cycle = itertools.cycle(API_KEYS)

def get_next_model(model_name: str):
    api_key = next(key_cycle)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

# -------------------
# Lazy-load FAISS
# -------------------
index = None
data = None

def build_index(data_list):
    texts = [d["english"] for d in data_list]
    embeddings = [genai.embed_content(model="models/embedding-001", content=t)["embedding"] for t in texts]
    embeddings = np.array(embeddings, dtype="float32")
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings)
    faiss.write_index(idx, "gita_index.faiss")
    return idx

def get_faiss_index():
    global index, data
    if index is not None and data is not None:
        return index, data

    with open("gita_verses.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    faiss_path = "gita_index.faiss"

    if os.path.exists(faiss_path):
        try:
            index = faiss.read_index(faiss_path)
            print("Loaded FAISS index from file ✅")
        except Exception as e:
            print("Error loading FAISS, rebuilding...", e)
            index = build_index(data)
    else:
        print("No FAISS file, building index...")
        index = build_index(data)

    return index, data

def find_relevant_verses(query, k=3):
    idx, data_list = get_faiss_index()
    embedding = genai.embed_content(model="models/embedding-001", content=query)["embedding"]
    query_embedding = np.array([embedding], dtype="float32")
    distances, indices = idx.search(query_embedding, k)
    gc.collect()
    return [data_list[i] for i in indices[0]]

# -------------------
# Routes
# -------------------
@app.route("/")
def home():
    random_num = random.randint(1, 15)
    image_file = f"images/{random_num}.jpg"
    return render_template("index.html", image_file=image_file)

@app.route("/ask-page")
def ask_page():
    random_num = random.randint(1, 15)
    image_file = f"images/{random_num}.jpg"
    return render_template("ask.html", image_file=image_file)

@app.route("/tech")
def tech():
    return render_template("tech.html")

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
        You are Lord Krishna, giving guidance to {name}, who is {age} years old.

        {name}'s Question:
        {query}
        
        Your task:
        1. Provide a compassionate, mentor-like answer appropriate for a {age}-year-old.
        2. Use all the three verses given and Whenever quoting verses:
        - Wrap the Sanskrit verse in <b> tags.
        - Wrap the translation in <b> tags as well.
        - Provide an explanation after each verse.
        3. Format the answer in paragraphs with spacing.
        4. Integrate verses contextually within the answer.
        5. At the end, optionally summarize the key points.

        Use a warm, divine, mentor tone.
        Knowledge base to reference (relevant slokas from Bhagavad Gita):
        {results}

        IMPORTANT:
        - KEEP THE LANGUAGE SIMPLE (English, Telugu, Hindi).
        - Output the response in HTML format.
        - Do NOT use Markdown-style asterisks.
        - Include Sanskrit and translation for every verse.
        - Maintain spacing between slokas.
        - Answer fully in {language}, go detailed and nicely.
        - End with a divine closing statement like:
          "I, Krishna, am always with you. Be strong."
        """

        llm_model = "gemini-2.5-pro" if mode == "deep" else "gemini-2.5-flash"
        llm = get_next_model(llm_model)
        response = llm.generate_content(prompt)
        krishna_response = response.text

        if krishna_response.startswith("```html"):
            krishna_response = krishna_response[7:]
        if krishna_response.endswith("```"):
            krishna_response = krishna_response[:-3]

        gc.collect()

        return jsonify({
            "response": krishna_response,
            "verses": results
        })

    except Exception as e:
        # Return JSON error to avoid frontend crash
        return jsonify({"error": str(e)}), 500

# -------------------
# Run App
# -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7100))
    app.run(host="0.0.0.0", port=port, debug=True)
