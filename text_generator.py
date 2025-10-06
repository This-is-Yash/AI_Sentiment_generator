# text_generator.py
# Generates text aligned with a given sentiment using GPT-2

from transformers import pipeline, set_seed

class SentimentTextGenerator:
    def __init__(self):
        # Load GPT-2 text generation pipeline
        self.generator = pipeline("text-generation", model="gpt2")
        set_seed(42)  # for reproducibility

    def generate_text(self, sentiment: str, prompt: str, max_length: int = 100):
        """
        Generate a paragraph aligned with the given sentiment.
        :param sentiment: 'positive' | 'negative' | 'neutral'
        :param prompt: user input
        :param max_length: length of output text
        """
        # Adjust the prompt slightly based on detected sentiment
        if sentiment == "positive":
            modified_prompt = f"{prompt}. The feeling is joyful and uplifting. "
        elif sentiment == "negative":
            modified_prompt = f"{prompt}. The tone is sad and pessimistic. "
        else:
            modified_prompt = f"{prompt}. The expression is neutral and objective. "

        # Generate text
        result = self.generator(
            modified_prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )[0]["generated_text"]

        return result


# 🔹 Example test
if __name__ == "__main__":
    gen = SentimentTextGenerator()
    sentiment = "positive"
    text = gen.generate_text(sentiment, "AI is transforming the future", max_length=80)
    print("\nGenerated Text:\n", text)
