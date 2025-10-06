# sentiment_model.py
# Prebuilt Sentiment Analysis using Hugging Face Transformers

from transformers import pipeline

class SentimentDetector:
    def __init__(self):
        # Pretrained English sentiment model (binary: pos/neg)
        self.pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def predict(self, text: str):
        """Return sentiment label ('positive'/'negative') and confidence score."""
        result = self.pipe(text[:512])[0]  # limit to 512 tokens
        label = result['label'].lower()    # e.g. 'positive' or 'negative'
        score = round(result['score'], 3)
        return label, score


# 🔹 Example test
if __name__ == "__main__":
    sd = SentimentDetector()
    s, conf = sd.predict("I really love working on AI projects!")
    print(f"Sentiment: {s}, confidence: {conf}")
