import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify

# Download VADER lexicon (needed once)
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# OCR.space API Key
OCR_API_KEY = "K84669025688957"

# Flask App
app = Flask(__name__)

# Extract text from image using OCR.space API
def extract_text(image_file):
    url = "https://api.ocr.space/parse/image"
    response = requests.post(
        url,
        files={"filename": image_file},
        data={"apikey": OCR_API_KEY, "language": "eng"},
    )
    
    result = response.json()
    
    if result.get("ParsedResults"):
        return result["ParsedResults"][0]["ParsedText"]
    return None

# Emotion classification
def classify_emotion(text):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.6:
        return "Happy"
    elif 0.2 <= compound < 0.6:
        return "Excited"
    elif -0.2 <= compound < 0.2:
        return "Neutral"
    elif -0.6 <= compound < -0.2:
        return "Sad"
    else:
        return "Angry"

# API Route: Upload image & analyze chat
@app.route("/analyze", methods=["POST"])
def analyze_chat():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    
    extracted_text = extract_text(image)

    if not extracted_text:
        return jsonify({"error": "No text found"}), 400

    messages = extracted_text.split("\n")
    results = [{"text": msg.strip(), "emotion": classify_emotion(msg.strip())} for msg in messages if msg.strip()]

    return jsonify({"status": "success", "data": results})

# Run API
if __name__ == "__main__":
    app.run(debug=True)
    