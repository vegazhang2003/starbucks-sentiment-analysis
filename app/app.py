import streamlit as st
from transformers import pipeline

st.title("Starbucks Review Sentiment Analysis")

st.write(
    "This app analyzes Starbucks customer reviews using Hugging Face models."
)

# Sentiment Analysis Pipeline
sentiment_pipeline = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Summarization Pipeline
summary_pipeline = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

review = st.text_area(
    "Enter a Starbucks customer review:",
    height=200
)

if st.button("Analyze"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        if len(review.split()) < 20:
            summary_text = review
        else:
            summary_result = summary_pipeline(
                review,
                max_length=30,
                min_length=8,
                do_sample=False
            )

            summary_text = summary_result[0]["summary_text"]

        sentiment_result = sentiment_pipeline(review)

        label = sentiment_result[0]["label"]
        score = sentiment_result[0]["score"]

        st.subheader("Review Summary")
        st.write(summary_text)

        st.subheader("Sentiment Result")

        if label == "POSITIVE":

            st.success("Positive Review")
            st.write(f"Confidence Score: {score:.2f}")

            st.info(
                "Business Suggestion: Maintain current service quality and customer experience."
            )

        else:

            st.error("Negative Review")
            st.write(f"Confidence Score: {score:.2f}")

            st.info(
                "Business Suggestion: Review customer complaints and improve service quality."
            )