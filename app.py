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

# Title and description
st.set_page_config(page_title="Movie Review Sentiment Analysis", layout="centered")
st.title("Movie Review Sentiment Analysis")

st.write("""
Analyze the sentiment of movie reviews using a machine learning model trained on the IMDB dataset.
Enter your review below to see if it's positive or negative!
""")

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Load the model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# Preprocess text
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

# Load model metrics
@st.cache_resource
def load_metrics():
    return joblib.load('model_metrics.pkl')

# Main function
model, vectorizer = load_model()
metrics = load_metrics()

# User input
user_input = st.text_area("Enter your movie review:", height=150, placeholder="Type your movie review here...")

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input
        processed_input = preprocess_text(user_input)
        
        # Vectorize the input
        input_vector = vectorizer.transform([processed_input])
        
        # Predict the sentiment
        prediction = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)[0]
        
        # Display the result
        st.markdown("### Results")
        
        # Display the sentiment with confidence
        if prediction == 1:
            confidence = prediction_proba[1]
            st.success(f"Positive Sentiment (Confidence: {confidence:.2f})")
        else:
            confidence = prediction_proba[0]
            st.error(f"Negative Sentiment (Confidence: {confidence:.2f})")
        
        # Display a confidence gauge
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Confidence Distribution")
            fig = px.pie(values=prediction_proba, 
                         names=["Negative", "Positive"], 
                         hole=0.4,
                         color_discrete_sequence=["#FF4B4B", "#0FFF50"])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Highlight keywords that influenced the prediction
            st.markdown("#### Top words in your review")
            # Create a small wordcloud of the input text
            wordcloud = WordCloud(width=400, 
                                height=200, 
                                background_color='white',
                                max_words=50,
                                colormap='viridis').generate(processed_input)
            
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        
        # Display model metrics
        with st.expander("Model Performance Metrics"):
            st.markdown(f"**Accuracy**: {metrics['accuracy']:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Positive Reviews**")
                st.write(f"Precision: {metrics['precision_pos']:.4f}")
                st.write(f"Recall: {metrics['recall_pos']:.4f}")
                st.write(f"F1-Score: {metrics['f1_pos']:.4f}")
            
            with col2:
                st.markdown("**Negative Reviews**")
                st.write(f"Precision: {metrics['precision_neg']:.4f}")
                st.write(f"Recall: {metrics['recall_neg']:.4f}")
                st.write(f"F1-Score: {metrics['f1_neg']:.4f}")
            
            # Display confusion matrix
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                      xticklabels=['Negative', 'Positive'], 
                      yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
    else:
        st.warning("Please enter a review.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Scikit-learn, and NLTK") 