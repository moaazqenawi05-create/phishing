from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

base_dir = os.path.dirname(__file__)

# تحميل الموديل
model = pickle.load(open(os.path.join(base_dir, "model.pkl"), "rb"))

@app.route('/')
def home():
    return "Server is working 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    text = data.get('message')

    if not text:
        return jsonify({"error": "No message provided"}), 400

    pred = model.predict([text])[0]

    return jsonify({
        "result": "spam" if pred == 1 else "safe"
    })

@app.route("/ping")
def ping():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)