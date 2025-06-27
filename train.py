# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load data
df = pd.read_csv("data/imdb.csv")
X = df['review']
y = df['sentiment']

# Train/test split
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save artifacts
os.makedirs("model", exist_ok=True)
joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(model, "model/model.pkl")

print("✅ Model training complete.")
