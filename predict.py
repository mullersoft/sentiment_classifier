# predict.py
import sys
import joblib
import numpy as np

def predict_sentiment(text):
    vectorizer = joblib.load("model/vectorizer.pkl")
    model = joblib.load("model/model.pkl")
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    confidence = np.max(model.predict_proba(features))
    return prediction, round(confidence, 2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your review text here\"")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    sentiment, confidence = predict_sentiment(input_text)
    print(f"Prediction: {sentiment} ({confidence} confidence)")
