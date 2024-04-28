# exploration of reviews
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
#pd.set_option('display.max_rows', 5000000)
#pd.set_option('display.max_columns', 5000000)

reviews = pd.read_csv("C:\\Users\\scott\\Downloads\\airbnb\\reviews.csv")
reviews.head()
reviews.info(verbose=True, show_counts=True)


listings = len(reviews.listing_id.unique())
days = len(reviews.date.unique())
days
listings

#drop 23 rows with NA comment values
reviews_cleaned = reviews.dropna(subset=['comments'])

### Sentiment analysis ###
from langdetect import detect, DetectorFactory

# Ensuring consistent results from langdetect
DetectorFactory.seed = 0

# Sample a subset of the data for language detection due to large dataset size
sentiment_df = reviews_cleaned['comments'].sample(n=1000, random_state=1)

# Detect language for the sample
def try_detect(text):
    try:
        return detect(text)
    except:
        return "unknown"
    
detected_languages = sentiment_df.apply(
    lambda x: "unknown" if x.strip() == "" else try_detect(x)
)

# View the distribution of languages detected in the sample
detected_languages.value_counts()

# Append the detected_languages Series to the sentiment_df DataFrame
sentiment_df = sentiment_df.to_frame().assign(language=detected_languages)

from transformers import pipeline, DistilBertTokenizer

# select model
sentiment_pipeline = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")

# Retrieve the data you want to analyze
data = sentiment_df['comments'].tolist()

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

# Function to truncate texts to the maximum sequence length
def truncate_texts(texts, max_length=300):
    truncated_texts = []
    for text in texts:
        # Encode the texts, truncating or padding to max_length
        # The tokenizer's encode_plus method automatically handles max_length
        # and truncates longer texts while ensuring special tokens are added.
        encoded_text = tokenizer.encode_plus(
            text, 
            max_length=max_length, 
            truncation=True, 
            add_special_tokens=True,  # Adds [CLS] and [SEP] tokens
            return_tensors='pt'  # Returns PyTorch tensors
        )
        # Convert the token IDs back to a string
        truncated_text = tokenizer.decode(encoded_text['input_ids'][0], skip_special_tokens=True)
        truncated_texts.append(truncated_text)
    return truncated_texts

data = truncate_texts(data)  # Truncate texts to fit within the model's limit

# Perform sentiment analysis
results = sentiment_pipeline(data)

# Extract sentiment labels and scores to separate lists
sentiment_labels = [result['label'] for result in results]
sentiment_scores = [result['score'] for result in results]

# Append these lists as new columns to your DataFrame
sentiment_df['sentiment_label'] = sentiment_labels
sentiment_df['sentiment_score'] = sentiment_scores