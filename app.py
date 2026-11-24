
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download English stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load trained model and TF-IDF vectorizer
model = joblib.load("best_fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return ' '.join(words)


# Page configuration
st.set_page_config(
    page_title="Smart Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Application heading
st.title("📰 Smart Fake News Detection System")

st.write(
    "This application uses Machine Learning "
    "to classify news as Fake or Real."
)

st.divider()

# Input fields
title = st.text_input("Enter News Title")

news = st.text_area(
    "Enter News Content",
    height=200
)

# Detect button
if st.button("Detect News"):

    if title.strip() and news.strip():

        # Combine title and news content
        combined_text = title + " " + news

        # Clean text
        cleaned_text = clean_text(combined_text)

        # Convert text into TF-IDF features
        text_vector = tfidf.transform([cleaned_text])

        # Make prediction
        prediction = model.predict(text_vector)[0]

        # Display result
        st.subheader("Prediction Result")

        if prediction == 0:
            st.error("⚠️ Likely Fake News")
        else:
            st.success("✅ Likely Real News")

        # Show prediction value
        st.write("Prediction Value:", int(prediction))

        # Show confidence score
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities) * 100

        st.write("Model Confidence:", round(confidence, 2), "%")

        # Show both class probabilities
        st.write("Fake News Probability:",
                 round(probabilities[0] * 100, 2), "%")

        st.write("Real News Probability:",
                 round(probabilities[1] * 100, 2), "%")

        # Show cleaned text for debugging
        with st.expander("View Cleaned Text"):
            st.write(cleaned_text)

        st.info(
            "Note: This prediction is based on patterns "
            "learned from the training dataset. "
            "It does not independently verify news facts."
        )

    else:
        st.warning(
            "Please enter both News Title and News Content."
        )

st.divider()

st.caption(
    "Major Project | Smart Fake News Detection System"
)
