import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.5-flash")

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

    prompt = f"You are an academic summarizer. Summarize the following text concisely and clearly in simple English. Focus on key points only.\n\n{text}"

    response = model.generate_content(prompt)
    summary = response.text

    return jsonify({"summary": summary.strip()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")