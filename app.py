# app.py
import streamlit as st
from sentiment_model import SentimentDetector
from text_generator import SentimentTextGenerator

# Initialize models
sentiment_model = SentimentDetector()
text_generator = SentimentTextGenerator()

# Streamlit UI
st.title("AI Sentiment Text Generator")
st.write("Enter a prompt and get a paragraph aligned with its sentiment.")

prompt = st.text_area("Your Prompt", "")

if st.button("Generate Text") and prompt.strip():
    # 1️⃣ Detect sentiment
    sentiment, score = sentiment_model.predict(prompt)
    
    # 2️⃣ Generate text based on sentiment
    generated_text = text_generator.generate_text(sentiment, prompt, max_length=150)
    
    # 3️⃣ Display results
    st.subheader("Detected Sentiment")
    st.write(f"{sentiment.capitalize()} (Confidence: {score})")
    
    st.subheader("Generated Text")
    st.write(generated_text)
