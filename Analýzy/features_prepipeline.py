import pandas as pd
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
from transformers import pipeline, DistilBertTokenizer
### REVIEWS ###

listings = pd.read_csv("C:\\Users\\scott\\Downloads\\listings.csv")
reviews = pd.read_csv("C:\\Users\\scott\\Downloads\\airbnb\\reviews.csv")
calendar = pd.read_csv("C:\\Users\\scott\\Downloads\\airbnb\\calendar.csv")

#drop 23 rows with NA comment values
reviews_cleaned = reviews.dropna(subset=['comments'])

sentiment_df = reviews_cleaned[['listing_id', 'comments']].sample(n=100, random_state=1)

sentiment = sentiment_df['comments'].tolist()

# select model
sentiment_pipeline = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")

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

data = truncate_texts(sentiment)  # Truncate texts to fit within the model's limit

# Perform sentiment analysis
results = sentiment_pipeline(data)

# Extract sentiment scores to separate lists
sentiment_scores = [result['score'] for result in results]

#append
sentiment_df['sentiment_score'] = sentiment_scores

reviews_final = sentiment_df.groupby(['listing_id'])['sentiment_score'].mean().reset_index()
#######################

### CALENDAR STATIC ###

calendar_new = calendar[['listing_id', 'available']]
calendar_new['busy'] = calendar_new.available.map( lambda x: 0 if x == 't' else 1)
# mean occupancy for each listing
calendar_static_final = calendar_new.groupby('listing_id')['busy'].mean().reset_index()

### CALENDAR TEMPORAL ###
#clean price
calendar['date'] = pd.to_datetime(calendar['date'])

def get_cleaned_price(price: pd.core.series.Series) -> float:
    """ Returns a float price from a pandas Series including the currency """
    return price.str.replace('$', '').str.replace(',', '').astype(float)

calendar['price'] = get_cleaned_price(calendar['price'])

# Creating 'in_season' dummy variable
calendar['in_season'] = calendar['date'].dt.month.isin([12, 4, 5, 6, 7, 8]).astype(int)

# Creating 'weekend' dummy variable
calendar['weekend'] = calendar['date'].dt.dayofweek.isin([4, 5, 6]).astype(int)

# Creating 'new_year' dummy variable
calendar['new_year'] = ((calendar['date'].dt.month == 12) & (calendar['date'].dt.day == 31)).astype(int)

calendar_dynamic_final = calendar.drop(['available', 'price', 'adjusted_price', 'minimum_nights', 'maximum_nights'], axis=1)
calendar_dynamic_final.head()