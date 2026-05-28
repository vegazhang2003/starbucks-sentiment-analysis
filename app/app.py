import streamlit as st
from transformers import pipeline

st.title("Starbucks Review Sentiment Analysis")

st.write(
    "This app analyzes Starbucks customer reviews using a Hugging Face sentiment classification model."
)

sentiment_pipeline = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
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

        label = sentiment_result[0]["label"]
        score = sentiment_result[0]["score"]

        st.subheader("Sentiment Result")

        if label == "POSITIVE":
            st.success("Positive Review")
            st.write(f"Confidence Score: {score:.2f}")
            st.info("Business Suggestion: Maintain current service quality and customer experience.")

        else:
            st.error("Negative Review")
            st.write(f"Confidence Score: {score:.2f}")
            st.info("Business Suggestion: Review customer complaints and improve service quality.")