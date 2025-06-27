
# IMDb Sentiment Classifier

A simple machine learning pipeline for classifying IMDb-style movie reviews as **positive** or **negative** using `scikit-learn`. This project uses TF-IDF vectorization and a Logistic Regression classifier.

---

## How to Download and Install Dependencies

Make sure you have Python 3 installed.

download the project:
```bash
git clone https://github.com/mullersoft/sentiment_classifier.git
```
Then install all required packages using:

```bash
pip install -r requirements.txt
```

---
## 1. Run as CMD

## Train the Model

To train the sentiment classifier using the dataset located at `data/imdb.csv`, run:

```bash
python train.py
```

This will generate:

- `model/model.pkl` — Trained logistic regression model  
- `model/vectorizer.pkl` — TF-IDF vectorizer

---

## Run Predictions

To classify a single review using the trained model, run:

```bash
python predict.py "This movie was absolutely amazing!"
```

**Example output:**
```
Prediction: positive (0.93 confidence)
```

---

##  2. Run as a Web API 

 wrap the model into a web API using Flask in `app.py`.

### Install Flask:
```bash
pip install flask
```
### Run the Flask app

```bash
python app.py
```
### Send a POST request

**With curl (Windows CMD compatible):**
```cmd
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"This movie was absolutely amazing!\"}"
```

**Example response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.93
}
```

---

## Project Structure

```
imdb_sentiment_classifier/
├── data/
│   └── imdb.csv                # Preprocessed dataset (~5000 samples)
├── model/
│   ├── model.pkl               # Trained classifier
│   └── vectorizer.pkl          # TF-IDF vectorizer
├── train.py                    # Trains and saves model/vectorizer
├── predict.py                  # Predicts sentiment from input text
├── app.py                      # Flask web server (optional)
├── requirements.txt            # Python dependencies
└── README.md                   # Project instructions
```
