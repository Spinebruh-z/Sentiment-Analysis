import streamlit as st
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

# Load the model and vectorizer
with open('model_xgb.pkl', 'rb') as file:
    model = pickle.load(file)

with open('countVectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Preprocessing function
def preprocess_review(review):
    stemmer = PorterStemmer()
    STOPWORDS = set(stopwords.words('english'))
    review = re.sub('[^a-zA-Z]', ' ', review)  # Keep only alphabets
    review = review.lower().split()  # Convert to lowercase and split
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]  # Stemming and stopword removal
    return ' '.join(review)

# Page configuration
st.set_page_config(page_title="Amazon Alexa Sentiment Analysis", layout="wide")

# App title with a header style
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #4CAF50;
        font-size: 36px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<div class="title">Amazon Alexa Sentiment Analysis</div>', unsafe_allow_html=True)

# Description
st.write(
    """
    This application predicts the sentiment of a review for Amazon Alexa products using a machine learning model.
    Enter your review below to find out whether it's **positive** or **negative** and see the confidence score.
    """
)

# Input field for user review
st.markdown("### Enter Your Review")
user_review = st.text_area("", placeholder="Type your review here...")

if st.button("Analyze Sentiment"):
    if user_review.strip():
        # Preprocess the input review
        preprocessed_review = preprocess_review(user_review)
        
        # Vectorize the input review
        vectorized_input = vectorizer.transform([preprocessed_review])
        
        # Predict sentiment
        prediction = model.predict(vectorized_input)
        prob = model.predict_proba(vectorized_input)[0]  # Get probabilities

        # Display sentiment
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        sentiment_color = "green" if sentiment == "Positive" else "red"
        
        st.markdown(
            f"""
            <h2 style="text-align: center; color: {sentiment_color};">
            Predicted Sentiment: {sentiment}
            </h2>
            """, unsafe_allow_html=True
        )
        
        # Visualization in two columns
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("### Sentiment Confidence")
            st.write("This chart represents the confidence levels for Positive and Negative sentiment predictions.")
        with col2:
            # Create a DataFrame for visualization
            sentiment_data = pd.DataFrame({
                'Sentiment': ['Negative', 'Positive'],
                'Probability': prob
            }).sort_values('Probability')  # Sort for better visualization
            st.bar_chart(sentiment_data.set_index('Sentiment'))
        
        # Add a subtle footer or call-to-action
        st.markdown(
            """
            ---
            **Want to learn more?** Check out Amazon Alexa reviews [here](https://www.amazon.com).
            """
        )
    else:
        st.warning("Please enter a review to analyze.")
