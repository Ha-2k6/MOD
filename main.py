import requests
import json
from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

OCR_API_KEY = "K84669025688957"  # Your OCR.space API key
analyzer = SentimentIntensityAnalyzer()  # Initialize sentiment analyzer

# Function to extract text from an image using OCR.space
def extract_text_from_image(image_path):
    with open(image_path, 'rb') as file:
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": file},
            data={"apikey": OCR_API_KEY, "language": "eng"}
        )
    
    result = response.json()
    
    if result.get("IsErroredOnProcessing"):
        return None
    return result["ParsedResults"][0]["ParsedText"]

# Function to predict emotion from text
def predict_emotion(text):
    sentiment_score = analyzer.polarity_scores(text)["compound"]
    
    if sentiment_score >= 0.3:
        return "happy"
    elif sentiment_score <= -0.3:
        return "angry"
    else:
        return "sad"

# API Endpoint to accept both text and image inputs
@app.route("/predict", methods=["POST"])
def predict():
    if "file" in request.files:  # If an image is uploaded
        file = request.files["file"]
        file_path = "temp_image.jpg"
        file.save(file_path)

        text = extract_text_from_image(file_path)
        if not text:
            return jsonify({"error": "Text extraction failed"}), 500

    elif "text" in request.json:  # If text is provided directly
        text = request.json["text"]
    else:
        return jsonify({"error": "No text or image provided"}), 400

    emotion = predict_emotion(text)
    
    return jsonify({"text": text, "emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)
  
