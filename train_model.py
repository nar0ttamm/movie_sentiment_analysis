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
import time

# Step 1: Download required NLTK data for text processing
print("Setting up NLTK...")
nltk.download('stopwords')  # Words like "the", "a", "an" that we want to ignore
nltk.download('punkt')      # For sentence tokenization

# Step 2: Connect to MongoDB (required for this project)
print("Connecting to MongoDB...")

# Function to connect with retry
def connect_to_mongodb(max_retries=3, retry_delay=2):
    retries = 0
    while retries < max_retries:
        try:
            # Try to connect to a local MongoDB server
            client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            # Test the connection
            client.server_info()
            print("Successfully connected to MongoDB!")
            
            # Set up database and collections
            db = client["movie_reviews_db"]
            reviews_collection = db["reviews"]
            model_collection = db["model_metrics"]
            user_reviews_collection = db["user_reviews"]
            
            return client, db, reviews_collection, model_collection, user_reviews_collection, True
        except Exception as e:
            retries += 1
            print(f"MongoDB connection attempt {retries} failed: {str(e)}")
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("All connection attempts failed.")
                return None, None, None, None, None, False

# Connect to MongoDB
client, db, reviews_collection, model_collection, user_reviews_collection, mongodb_available = connect_to_mongodb()

if not mongodb_available:
    print("\n==================================================================")
    print("ERROR: MongoDB connection failed. This project requires MongoDB.")
    print("Please make sure MongoDB is installed and running on localhost:27017")
    print("Install MongoDB using:")
    print("  sudo apt update")
    print("  sudo apt install -y mongodb")
    print("  sudo systemctl start mongodb")
    print("==================================================================\n")
    sys.exit(1)  # Exit the script if MongoDB is not available

# Step 3: Load the movie reviews dataset
print("Loading dataset...")
df = pd.read_csv("IMDB Dataset.csv")
print(f"Loaded {len(df)} reviews from IMDB Dataset")

# Step 4: Define a function to clean and prepare the text
def clean_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', ' ', text)
    # Keep only letters (remove numbers, punctuation)
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Split into words
    words = text.split()
    # Remove common words like "the", "is", etc.
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Reduce words to their root form
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Join the words back together
    return ' '.join(words)

# Step 5: Clean all the reviews
print("Cleaning and preprocessing text data...")
df['processed_review'] = df['review'].apply(clean_text)

# Step 6: Convert sentiment labels to numbers (1 for positive, 0 for negative)
df['sentiment_label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Step 7: Store all IMDB reviews in MongoDB
print("Storing all IMDB reviews in MongoDB...")
# Clear existing data first
reviews_collection.delete_many({})
# Convert the dataframe to a list of dictionaries
reviews_data = df[['review', 'processed_review', 'sentiment', 'sentiment_label']].to_dict('records')
# Insert into MongoDB
result = reviews_collection.insert_many(reviews_data)
print(f"Successfully inserted {len(result.inserted_ids)} reviews into MongoDB")

# Step 8: Check for any user reviews in the database
print("Checking for user reviews in MongoDB...")
user_reviews = list(user_reviews_collection.find({}))
if user_reviews:
    print(f"Found {len(user_reviews)} user reviews in the database")
    
    # Convert user reviews to dataframe
    user_df = pd.DataFrame(user_reviews)
    
    # Create a combined dataframe with both IMDB and user reviews
    user_df = user_df[['review', 'processed_review', 'prediction', 'confidence']]
    user_df['sentiment'] = user_df['prediction']
    user_df['sentiment_label'] = user_df['prediction'].apply(lambda x: 1 if x == 'positive' else 0)
    
    # Combine datasets (optional - if you want to include user reviews in training)
    # Uncomment to include user reviews in your training
    # df = pd.concat([df, user_df[['review', 'processed_review', 'sentiment', 'sentiment_label']]], ignore_index=True)
    # print(f"Combined dataset now has {len(df)} reviews")
else:
    print("No user reviews found in the database")

# Step 9: Get all reviews from MongoDB to ensure we're using the database
print("Retrieving all reviews from MongoDB for training...")
mongo_reviews = list(reviews_collection.find({}))
print(f"Retrieved {len(mongo_reviews)} reviews from MongoDB")

# Convert to DataFrame
mongo_df = pd.DataFrame(mongo_reviews)

# Step 10: Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    mongo_df['processed_review'],  # Use the data from MongoDB
    mongo_df['sentiment_label'],   # The sentiment labels from MongoDB
    test_size=0.2,                 # Use 20% for testing
    random_state=42                # Set seed for reproducibility
)

# Step 11: Convert text to numerical features using TF-IDF
print("Converting text to numerical features...")
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features for efficiency
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Step 12: Train the logistic regression model
print("Training model...")
model = LogisticRegression(max_iter=1000)  # Increase iterations to ensure convergence
model.fit(X_train_vectors, y_train)

# Step 13: Evaluate the model on test data
print("Evaluating model performance...")
y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# Step 14: Print the performance metrics
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 15: Save the metrics to MongoDB
print("Saving metrics to MongoDB...")
metrics = {
    "accuracy": accuracy,
    "precision_pos": report['1']['precision'],
    "recall_pos": report['1']['recall'],
    "f1_pos": report['1']['f1-score'],
    "precision_neg": report['0']['precision'],
    "recall_neg": report['0']['recall'],
    "f1_neg": report['0']['f1-score'],
    "confusion_matrix": cm.tolist(),
    "last_updated": pd.Timestamp.now().isoformat()
}
model_collection.delete_many({})  # Clear existing data
model_collection.insert_one(metrics)

# Step 16: Create and save a visualization of the confusion matrix
print("Creating confusion matrix visualization...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Step 17: Save the model and vectorizer for later use
print("Saving model and vectorizer...")
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Step 18: Save model metrics to a file (as backup)
print("Saving metrics to file...")
metrics_file = {
    "accuracy": float(accuracy),
    "precision_pos": float(report['1']['precision']),
    "recall_pos": float(report['1']['recall']),
    "f1_pos": float(report['1']['f1-score']),
    "precision_neg": float(report['0']['precision']),
    "recall_neg": float(report['0']['recall']),
    "f1_neg": float(report['0']['f1-score']),
    "confusion_matrix": cm.tolist(),
    "last_updated": pd.Timestamp.now().isoformat()
}
joblib.dump(metrics_file, 'model_metrics.pkl')

# Step 19: Print MongoDB stats
print("\nMongoDB Database Statistics:")
print(f"IMDB Reviews: {reviews_collection.count_documents({})}")
print(f"User Reviews: {user_reviews_collection.count_documents({})}")
print(f"Model Metrics: {model_collection.count_documents({})}")

print("\nTraining completed successfully!")
print("You can now run the Streamlit app using: streamlit run app.py") 