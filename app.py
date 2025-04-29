import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import pymongo

# Set up the page
st.set_page_config(page_title="Movie Review Sentiment Analysis", layout="centered")
st.title("Movie Review Sentiment Analysis")

# App description
st.write("""
Analyze the sentiment of movie reviews using a machine learning model trained on the IMDB dataset.
Enter your review below to see if it's positive or negative!
""")

# Make sure we have the required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Connect to MongoDB
@st.cache_resource
def connect_to_mongodb():
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        # Test the connection
        client.server_info()
        db = client["movie_reviews_db"]
        return db, True
    except Exception as e:
        st.warning(f"MongoDB connection failed: {str(e)}. Some features will be limited.")
        return None, False

# Load our trained model and vectorizer
@st.cache_resource  # This prevents reloading on each run
def load_model():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# Clean up and prepare text for the model
def clean_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', ' ', text)
    # Keep only letters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Make everything lowercase
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

# Load model performance metrics from MongoDB or file
@st.cache_data
def load_metrics():
    db, mongodb_available = connect_to_mongodb()
    
    if mongodb_available:
        # Try to get metrics from MongoDB first
        metrics = db["model_metrics"].find_one()
        if metrics:
            return metrics
    
    # Fallback to file
    return joblib.load('model_metrics.pkl')

# Safer word cloud generation function
def create_word_cloud(text):
    # Add default words to prevent empty word clouds
    default_text = "movie review sentiment analysis film cinema"
    
    # Check if text is empty or very short
    if not text or len(text.strip()) < 3:
        text = default_text
    else:
        # Ensure there are enough words by adding defaults
        text = text + " " + default_text
    
    try:
        # Create word cloud with the combined text
        wordcloud = WordCloud(
            width=400,
            height=200,
            background_color='white',
            max_words=50,
            min_word_length=2,
            colormap='viridis',
            prefer_horizontal=0.9,
            regexp=r'\w[\w\']+'
        ).generate(text)
        
        # Display the word cloud
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        # Create a simple figure with error message
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"Word cloud error: {str(e)}", ha='center', va='center')
        ax.axis('off')
        return fig

# Store a user review in MongoDB
def store_review_in_mongodb(review_text, cleaned_text, prediction, confidence):
    db, mongodb_available = connect_to_mongodb()
    if mongodb_available:
        try:
            # Prepare the review data
            review_data = {
                "review": review_text,
                "processed_review": cleaned_text,
                "prediction": "positive" if prediction == 1 else "negative",
                "confidence": float(confidence),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            # Store in user_reviews collection
            db["user_reviews"].insert_one(review_data)
            return True
        except Exception as e:
            st.error(f"Failed to store review: {str(e)}")
            return False
    return False

# Try to load the model and metrics
try:
    model, vectorizer = load_model()
    metrics = load_metrics()
    model_ready = True
except Exception as e:
    st.error(f"Problem loading model: {str(e)}")
    model_ready = False

# Text input area for the review
user_review = st.text_area(
    "Enter your movie review:", 
    height=150, 
    placeholder="Type your movie review here..."
)

# When the analyze button is clicked
if st.button("Analyze Sentiment"):
    # Check if user entered any text
    if not user_review or len(user_review.strip()) == 0:
        st.warning("Please enter a review before analyzing.")
    elif not model_ready:
        st.error("Model is not loaded. Please check if model files exist.")
    else:
        # Clean the text
        cleaned_review = clean_text(user_review)
        
        # Check if we have any words left after cleaning
        if not cleaned_review or len(cleaned_review.strip()) == 0:
            st.warning("After removing common words, no meaningful text remained. Please enter a more detailed review.")
        else:
            # Convert text to numbers the model can understand
            review_features = vectorizer.transform([cleaned_review])
            
            # Make a prediction
            prediction = model.predict(review_features)[0]
            confidence_scores = model.predict_proba(review_features)[0]
            
            # Show the results
            st.markdown("### Results")
            
            # Show the sentiment prediction with confidence
            if prediction == 1:  # Positive sentiment
                confidence = int(confidence_scores[1] * 100)
                st.success(f"Positive Sentiment (Confidence: {confidence}%)")
            else:  # Negative sentiment
                confidence = int(confidence_scores[0] * 100)
                st.error(f"Negative Sentiment (Confidence: {confidence}%)")
            
            # Store the review in MongoDB
            confidence_value = confidence_scores[1] if prediction == 1 else confidence_scores[0]
            saved = store_review_in_mongodb(user_review, cleaned_review, prediction, confidence_value)
            if saved:
                st.info("✅ Review saved to MongoDB database")
                
            # Display the results in two columns
            col1, col2 = st.columns([1, 1])
            
            # First column: Confidence pie chart
            with col1:
                st.markdown("#### Confidence Distribution")
                fig = px.pie(
                    values=confidence_scores, 
                    names=["Negative", "Positive"], 
                    hole=0.4,
                    color_discrete_sequence=["#FF4B4B", "#0FFF50"]
                )
                # Show percentages on the chart
                fig.update_traces(textinfo='percent')
                st.plotly_chart(fig, use_container_width=True)
            
            # Second column: Word cloud
            with col2:
                st.markdown("#### Top words in your review")
                word_cloud_fig = create_word_cloud(cleaned_review)
                st.pyplot(word_cloud_fig)
            
            # Show model metrics in expandable section
            with st.expander("Model Performance Metrics"):
                st.markdown(f"**Accuracy**: {int(metrics['accuracy']*100)}%")
                
                # Display metrics in two columns
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Positive Reviews**")
                    st.write(f"Precision: {int(metrics['precision_pos']*100)}%")
                    st.write(f"Recall: {int(metrics['recall_pos']*100)}%")
                    st.write(f"F1-Score: {int(metrics['f1_pos']*100)}%")
                
                with col2:
                    st.markdown("**Negative Reviews**")
                    st.write(f"Precision: {int(metrics['precision_neg']*100)}%")
                    st.write(f"Recall: {int(metrics['recall_neg']*100)}%")
                    st.write(f"F1-Score: {int(metrics['f1_neg']*100)}%")
                
                # Show confusion matrix
                cm = np.array(metrics["confusion_matrix"])
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive']
                )
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

# Show MongoDB status
db, mongodb_available = connect_to_mongodb()
if mongodb_available:
    try:
        # Count reviews in MongoDB
        imdb_count = db["reviews"].count_documents({})
        user_count = db["user_reviews"].count_documents({})
        
        # Add a footer with MongoDB status
        st.markdown("---")
        st.markdown(f"**MongoDB Status**: Connected ✅")
        st.markdown(f"**Database Statistics**: {imdb_count} IMDB reviews, {user_count} user reviews stored")
    except:
        st.markdown("---")
        st.markdown("**MongoDB Status**: Connected but error accessing collections ⚠️")
else:
    st.markdown("---")
    st.markdown("**MongoDB Status**: Not connected ❌")
