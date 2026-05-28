import streamlit as st
from transformers import pipeline

st.title("Starbucks Review Sentiment Analysis")

st.write(
    "This app analyzes Starbucks customer reviews using two Hugging Face pipelines: "
    "sentiment classification and short business suggestion generation."
)

# Pipeline 1: Sentiment classification
sentiment_pipeline = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Pipeline 2: Text generation for business suggestion
suggestion_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
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

        suggestion_prompt = (
            "Give one short business suggestion for this Starbucks customer review: "
            + review
        )

        suggestion_result = suggestion_pipeline(
            suggestion_prompt,
            max_new_tokens=40
        )

        suggestion = suggestion_result[0]["generated_text"]

        st.subheader("Sentiment Result")

        if label == "POSITIVE":
            st.success("Positive Review")
        else:
            st.error("Negative Review")

        st.write(f"Confidence Score: {score:.2f}")

        st.subheader("Business Suggestion")
        st.info(suggestion)