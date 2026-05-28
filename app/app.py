import streamlit as st
from transformers import pipeline

st.title("Starbucks Review Sentiment Analysis")

st.write(
    "This app analyzes Starbucks customer reviews using two Hugging Face pipelines: "
    "sentiment classification and emotion classification."
)

sentiment_pipeline = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

emotion_pipeline = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion"
)

review = st.text_area(
    "Enter a Starbucks customer review:",
    height=200
)

if st.button("Analyze"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        sentiment_result = sentiment_pipeline(review)
        emotion_result = emotion_pipeline(review)

        sentiment_label = sentiment_result[0]["label"]
        sentiment_score = sentiment_result[0]["score"]

        emotion_label = emotion_result[0]["label"]
        emotion_score = emotion_result[0]["score"]

        st.subheader("Sentiment Result")

        if sentiment_label == "POSITIVE":
            st.success("Positive Review")
            suggestion = "Maintain current service quality and customer experience."
        else:
            st.error("Negative Review")
            suggestion = "Review customer complaints and improve service quality."

        st.write(f"Confidence Score: {sentiment_score:.2f}")

        st.subheader("Emotion Result")
        st.write(f"Detected Emotion: {emotion_label}")
        st.write(f"Emotion Confidence Score: {emotion_score:.2f}")

        st.subheader("Business Suggestion")
        st.info(suggestion)