import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define labels for sentiment
labels = ['negative', 'neutral', 'positive']


def get_sentiment(text):
    # Ensure the input text is a string
    text = str(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits[0].numpy()
    sentiment = labels[np.argmax(scores)]
    return sentiment


# Load your dataset
df = pd.read_csv("combined_tweets.csv")

# Ensure all tweets are strings and handle missing values
df['Tweet'] = df['Tweet'].fillna('').astype(str)

# Apply the sentiment analysis function
df['Sentiment'] = df['Tweet'].apply(get_sentiment)

# Save the output to a new CSV file
output_file = "sentiment_output.csv"
df.to_csv(output_file, index=False)

print(f"Sentiment analysis results saved to {output_file}")
