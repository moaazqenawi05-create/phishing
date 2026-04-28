from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

import os

base_dir = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(base_dir, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(base_dir, "vectorizer.pkl"), "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    return jsonify({
        "result": "spam" if pred == 1 else "safe"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)