from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load models (صح 100%)
text_model = joblib.load("text_model.pkl")
url_model = joblib.load("url_model.pkl")


# detect URL
def is_url(text):
    return "http" in text or "www" in text


@app.route("/")
def home():
    return "🚀 Hybrid ML API is running"


# ---------------------------
# TEXT MODEL ENDPOINT
# ---------------------------
@app.route("/text_predict", methods=["POST"])
def text_predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No input provided"}), 400

    text = data["text"]

    pred = text_model.predict([text])[0]

    result = "spam" if pred == 1 else "ham"

    return jsonify({"result": result})


# ---------------------------
# URL MODEL ENDPOINT
# ---------------------------
@app.route("/url_predict", methods=["POST"])
def url_predict():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "No features provided"}), 400

    features = data["features"]  # لازم تبقى list فيها 30 قيمة

    pred = url_model.predict([features])[0]

    result = "phishing" if pred in [-1, 1] else "safe"

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)