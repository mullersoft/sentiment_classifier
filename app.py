from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Please provide a 'text' field in JSON body."}), 400

    text = data["text"]
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    confidence = np.max(model.predict_proba(features))

    return jsonify({
        "sentiment": prediction,
        "confidence": round(float(confidence), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
