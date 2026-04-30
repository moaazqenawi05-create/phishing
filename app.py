from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib   # ✅ بدل pickle
import os

app = Flask(__name__)
CORS(app)

base_dir = os.path.dirname(__file__)

# ✅ تحميل الموديل صح
model_path = os.path.join(base_dir, "spam_model.pkl")
model = joblib.load(model_path)

@app.route('/')
def home():
    return "Server is working 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400

    text = data['message']

    pred = model.predict([text])[0]

    return jsonify({
        "result": "spam" if pred == 1 else "safe"
    })

@app.route("/ping")
def ping():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)