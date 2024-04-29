import pandas as pd
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
from transformers import pipeline, DistilBertTokenizer

def merge_datasets_and_save(first_csv_path, second_csv_path, output_csv_path):
    """
    1)BASIC INFO:
    VE FUNKCI MĚŇTE POUZE NÁZEV SLOUPCE TABULKY ADDITIONAL INFO, KTERÝ VSTOUPÍ NA JOIN (additional_info.rename(columns={'listing_id': 'id'}),
    NÁZEV SLOUPCE NA JOIN  (merged_df = pd.merge(model_pop, additional_info, on='id', how='left'))
    A POČET ZÁZNAMŮ MODELOVÉ POPULACE PODLE POTŘEBY (model_pop = model_pop.iloc[:1000])
    V ČÁSTI def create_features(merged_df) MŮŽETE KODIT CO POTŘEBUJETE PODLE INSTRUKCÍ 
    PO VŠECH ÚPRAVÁCH ZAVOLEJTE FUNKCI S JEJÍMI PARAMETRY - SEKCE 3
    
    2)POPIS FUNKCE:
    Načte dva CSV soubory model_pop(calendar) a additional_info(listings, reviews), přejmenuje sloupec v additional_info dle potřeby, provede left join model_pop 
    (omezeného na prvních X záznamů pokud chcete) s additional_info na základě zadaného sloupce(zde listing_id)
    a uloží výsledek do výsledného features CSV souboru.

    3)PARAMETRY:
    first_csv_path (str): Cesta k model_pop.
    second_csv_path (str): Cesta k additional_info 
    output_csv_path (str): Cesta, kam se má výsledný CSV soubor uložit (features dataframe)
    """

    merged_df = pd.DataFrame()

    def create_features(merged_df):
        # ZDE MŮŽETE PROVÁDĚT CO CHCETE SE SVÝMI DATY
        # HLAVNĚ NECHTE NÁZEV VÝSLEDNÉHO DATAFRAME JAKO merged_df, ABY SE KOD NEMUSEL MĚNIT A PROBĚHL TAK, JAK JE NYNÍ
        #drop 23 rows with NA comment values
            # Načtení prvního datasetu a omezení na prvních X záznamů - X si volte jak chcete
        model_pop = pd.read_csv(first_csv_path)  

        # Načtení druhého datasetu a přejmenování sloupce
        additional_info = pd.read_csv(second_csv_path)
        #additional_info = additional_info.iloc[:500]

        reviews = additional_info
        reviews_cleaned = reviews.dropna(subset=['comments'])

        sentiment_df = reviews_cleaned[['listing_id', 'comments']]

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

        additional_info = sentiment_df.groupby(['listing_id'])['sentiment_score'].mean().reset_index()

        model_pop = model_pop.drop(['date', 'available', 'price', 'adjusted_price', 'minimum_nights', 'maximum_nights'], axis=1)    
        
        # Provedení left join prvního datasetu s druhým datasetem
        merged_df = pd.merge(model_pop, additional_info, on='listing_id', how='left')
        merged_df = merged_df.groupby('listing_id').sample(n=1)
        
        return merged_df

    features = create_features(merged_df)

    # Uložení výsledného datasetu do CSV souboru
    features.to_csv(output_csv_path, index=False)
    
    print(f"Data byla úspěšně spojena a uložena do souboru {output_csv_path}")

merge_datasets_and_save("DataX---Banger-Team\\Data\\calendar.csv", "DataX---Banger-Team\\Data\\reviews.csv", "DataX---Banger-Team\\Data\\features_cal_rev.csv")
