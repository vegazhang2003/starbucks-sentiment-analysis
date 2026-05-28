import streamlit as st
from transformers import pipeline

st.title("Starbucks Review Sentiment Analysis")

st.write(
    "This app analyzes Starbucks customer reviews using two Hugging Face pipelines: "
    "review summarization and sentiment classification."
)

# Pipeline 1: Sentiment Analysis using fine-tuned DistilBERT
sentiment_pipeline = pipeline(
    "text-classification",
    model="../models/starbucks_sentiment_model"
)

# Pipeline 2: Summarization using Hugging Face pre-trained model
summary_pipeline = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

review = st.text_area("Enter a Starbucks customer review:")

if st.button("Analyze"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        # For very short reviews, summarization may not work well.
        if len(review.split()) < 20:
            summary_text = review
        else:
            summary_result = summary_pipeline(
                review,
                max_length=35,
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

        if label == "LABEL_1":
            st.write("Prediction: Positive")
            st.write(f"Confidence: {score:.2f}")
            st.success("Positive Review")
            st.info("Business Suggestion: Maintain current service quality and customer experience.")
        else:
            st.write("Prediction: Negative")
            st.write(f"Confidence: {score:.2f}")
            st.error("Negative Review")
            st.info("Business Suggestion: Review customer complaints and improve service quality.")