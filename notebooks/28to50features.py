import pandas as pd
import re
import json
import numpy as np

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
    # Načtení prvního datasetu a omezení na prvních X záznamů - X si volte jak chcete
    model_pop = pd.read_csv(first_csv_path)
    #model_pop = model_pop.iloc[:500]  

    # Načtení druhého datasetu a přejmenování sloupce
    additional_info = pd.read_csv(second_csv_path)
    additional_info.rename(columns={'id': 'listing_id'}, inplace=True)

    # Provedení left join prvního datasetu s druhým datasetem
    merged_df = pd.merge(model_pop, additional_info, on='listing_id', how='left')
    merged_df = merged_df.groupby('listing_id').sample(n=1)

    def create_features(merged_df):
         #reviews.groupby(['listing_id'])['sentiment_score'].mean().reset_index()
         
        merged_df=merged_df[["listing_id","neighbourhood_cleansed","room_type","accommodates","bathrooms_text","beds","amenities"]]

        def extract_shared(text):
            if pd.isnull(text):
                return 0  # Return 0 if the value is NaN
            elif 'shared' in text.lower():
                return 1  # Return 1 if the word "shared" is found
            else:
                return 0  # Return 0 otherwise
        merged_df["bathrooms_shared"] = merged_df['bathrooms_text'].apply(extract_shared)
        merged_df["bathrooms_shared"] = merged_df['bathrooms_shared'].astype('category')

        #Getting the numeric values from bathrooms_text
        def extract_numerical_value(text):
        # Regular expression pattern to match numerical values
            pattern = r'(\d+(\.\d+)?)'  # Matches one or more digits, possibly followed by a decimal point and more digits
            matches = re.findall(pattern, str(text))
            if matches:
                return float(matches[0][0])  # Extracting the numerical value and converting it to float
            else:
                return np.nan  # Return NaN if no numerical value is found

        merged_df["bathrooms_numeric"] = merged_df['bathrooms_text'].apply(extract_numerical_value)
        merged_df["bathrooms_numeric"] = merged_df["bathrooms_numeric"].astype(float)
        
        # Unified neighbourhoods
        mapping = {
            'Velká Chuchle': 'Praha 16',
            'Dolní Chabry': 'Praha 8',
            'Kunratice': 'Praha 4',
            'Zličín': 'Praha 17',
            'Dubeč': 'Praha 15',
            'Zbraslav': 'Praha 16',
            'Petrovice': 'Praha 15',
            'Suchdol': 'Praha 6',
            'Klánovice': 'Praha 21',
            'Šeberov': 'Praha 11',
            'Újezd': 'Praha 11',
            'Štěrboholy': 'Praha 15',
            'Kolovraty': 'Praha 22',
            'Řeporyje': 'Praha 13',
            'Ďáblice': 'Praha 8',
            'Slivenec': 'Praha 5',
            'Dolní Počernice': 'Praha 14',
            'Koloděje': 'Praha 21',
            'Březiněves': 'Praha 8',
            'Nebušice': 'Praha 6',
            'Satalice': 'Praha 19',
            'Čakovice': 'Praha 18',
            'Lipence': 'Praha 16',
            'Dolní Měcholupy': 'Praha 15',
            'Vinoř': 'Praha 19',
            'Nedvězí': 'Praha 22',
            'Přední Kopanina': 'Praha 6',
            'Lysolaje': 'Praha 6',
            'Libuš': 'Praha 12',
            'Troja': 'Praha 7'
        }
        # Iterate over items in the mapping dictionary and replace values
        merged_df['neighbourhood_cleansed'] = merged_df['neighbourhood_cleansed'].apply(lambda x: mapping[x] if x in mapping else x)

        # Create a column which contains the number of amenities for each listing
        merged_df["amenities_num"] = merged_df["amenities"].apply(len)

        # Fill missing values with the median
        median_value = merged_df['bathrooms_numeric'].median()
        merged_df['bathrooms_numeric'] = merged_df['bathrooms_numeric'].fillna(median_value)
        median_value = merged_df['beds'].median()
        merged_df['beds'] = merged_df['beds'].fillna(median_value)
        
        #Poslední merge
        merged_df=merged_df[["listing_id","neighbourhood_cleansed", "room_type","accommodates","bathrooms_numeric","beds", "bathrooms_shared", "amenities_num"]]
        return merged_df

    features = create_features(merged_df)

    # Uložení výsledného datasetu do CSV souboru
    features.to_csv(output_csv_path, index=False)
    
    print(f"Data byla úspěšně spojena a uložena do souboru {output_csv_path}")

merge_datasets_and_save("Data\\raw\\calendar.csv", "Data\\raw\\listings.csv", "Data\\interim\\features-j-2.csv")
