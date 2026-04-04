import os
from flask import Flask, request, jsonify, render_template
from groq import Groq
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an academic summarizer. Summarize the given text concisely and clearly in simple English. Focus on the key points only."
            },
            {
                "role": "user",
                "content": f"Summarize this:\n\n{text}"
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    summary = chat_completion.choices[0].message.content

    return jsonify({"summary": summary.strip()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")