import os
os.environ["HF_HOME"] = "E:\\huggingface_cache"

from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import PyPDF2

app = Flask(__name__)

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    text = ""

    if "file" in request.files:
        file = request.files["file"]
        if file.filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        elif file.filename.endswith(".txt"):
            text = file.read().decode("utf-8")
    else:
        data = request.json
        text = data["text"]

    if not text.strip():
        return jsonify({"error": "No text found"}), 400

    # Split text if too long
    max_chunk = 1000
    words = text.split()
    chunks = [" ".join(words[i:i+max_chunk]) for i in range(0, len(words), max_chunk)]
     
    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summary += result[0]["summary_text"] + " "

    return jsonify({"summary": summary.strip()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")


