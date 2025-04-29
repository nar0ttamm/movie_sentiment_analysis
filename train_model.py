import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pymongo
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Connect to MongoDB with error handling
try:
    # Try localhost connection first (MongoDB Community)
    client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    # Verify connection
    client.server_info()
    print("Connected to local MongoDB instance")
except pymongo.errors.ServerSelectionTimeoutError:
    print("Warning: Could not connect to local MongoDB. Make sure MongoDB is running.")
    print("Saving model without MongoDB storage.")
    client = None

# Set up database and collections if MongoDB is available
if client:
    db = client["movie_reviews_db"]
    reviews_collection = db["reviews"]
    model_collection = db["model_metrics"]

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("IMDB Dataset.csv")

# Data Preprocessing
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', ' ', text)
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Join tokens back into text
    return ' '.join(tokens)

print("Preprocessing data...")
df['processed_review'] = df['review'].apply(preprocess_text)

# Convert sentiment labels to binary
df['sentiment_label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Store preprocessed data in MongoDB if available
if client:
    print("Storing preprocessed data in MongoDB...")
    reviews_collection.delete_many({})  # Clear existing data
    reviews_data = df[['review', 'processed_review', 'sentiment', 'sentiment_label']].to_dict('records')
    reviews_collection.insert_many(reviews_data)
else:
    print("Skipping MongoDB storage due to connection issue.")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_review'], 
    df['sentiment_label'],
    test_size=0.2,
    random_state=42
)

# Feature extraction with TF-IDF
print("Extracting features...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train the model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectors, y_train)

# Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save metrics to MongoDB if available
if client:
    metrics = {
        "accuracy": accuracy,
        "precision_pos": report['1']['precision'],
        "recall_pos": report['1']['recall'],
        "f1_pos": report['1']['f1-score'],
        "precision_neg": report['0']['precision'],
        "recall_neg": report['0']['recall'],
        "f1_neg": report['0']['f1-score'],
        "confusion_matrix": cm.tolist()
    }
    model_collection.delete_many({})  # Clear existing data
    model_collection.insert_one(metrics)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Save model and vectorizer
print("Saving model and vectorizer...")
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Also save metrics to a file to ensure availability even without MongoDB
metrics_file = {
    "accuracy": float(accuracy),
    "precision_pos": float(report['1']['precision']),
    "recall_pos": float(report['1']['recall']),
    "f1_pos": float(report['1']['f1-score']),
    "precision_neg": float(report['0']['precision']),
    "recall_neg": float(report['0']['recall']),
    "f1_neg": float(report['0']['f1-score']),
    "confusion_matrix": cm.tolist()
}
joblib.dump(metrics_file, 'model_metrics.pkl')

print("Training completed successfully!") 