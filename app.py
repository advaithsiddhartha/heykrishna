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

# -------------------
# Setup Flask + dotenv
# -------------------
app = Flask(__name__)
load_dotenv()

# -------------------
# Load FAISS + Gita Data
# -------------------
index = faiss.read_index("gita_index.faiss")
with open("gita_verses.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# -------------------
# API Key Rotation Setup
# -------------------
api_keys_str = os.getenv("API_KEYS", "")
API_KEYS = [k.strip() for k in api_keys_str.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("No API keys found. Please set API_KEYS in your .env or hosting environment.")

key_cycle = itertools.cycle(API_KEYS)

def get_next_model(model_name: str):
    api_key = next(key_cycle)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

# -------------------
# Utility: Search Verses
# -------------------
def find_relevant_verses(query, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [data[i] for i in indices[0]]

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
    req = request.json
    name = req.get("name", "Friend")
    age = req.get("age", "unknown")
    query = req.get("query", "")
    language = req.get("language", "English")
    mode = req.get("mode", "normal")  # normal / deep

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
        - Output the response in **HTML format**.
        - Do NOT use Markdown-style asterisks. Use <b> tags for bold text.
        - Include Sanskrit and translation for every verse.
        - Maintain spacing between slokas.
        - Answer fully in {language}, go detailed and nicely.
        - No need of bullet points or summaries. End with a divine closing statement like:
          "I, Krishna, am always with you. Be strong."
    """

    # Select model with rotation
    if mode == "deep":
        llm = get_next_model("gemini-2.5-pro")
    else:
        llm = get_next_model("gemini-2.5-flash")

    response = llm.generate_content(prompt)

    krishna_response = response.text
    if krishna_response.startswith("```html"):
        krishna_response = krishna_response[7:]
    if krishna_response.endswith("```"):
        krishna_response = krishna_response[:-3]

    return jsonify({
        "response": krishna_response,
        "verses": results
    })

# -------------------
# Run App
# -------------------
if __name__ == "__main__":
    app.run(debug=True, port=8100)
