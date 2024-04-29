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
            # Načtení prvního datasetu a omezení na prvních X záznamů - X si volte jak chcete
        model_pop = pd.read_csv(first_csv_path)
        #model_pop = model_pop.iloc[:500]  

        # Načtení druhého datasetu a přejmenování sloupce
        additional_info = pd.read_csv(second_csv_path)
        #additional_info.rename(columns={'id': 'listing_id'}, inplace=True)

        ### CALENDAR STATIC ###

        calendar_new = additional_info[['listing_id', 'available']]
        calendar_new['busy'] = calendar_new.available.map( lambda x: 0 if x == 't' else 1)
        # mean occupancy for each listing
        additional_info = calendar_new.groupby('listing_id')['busy'].mean().reset_index()

        ### CALENDAR TEMPORAL ###
        #clean price
        calendar = model_pop
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

        model_pop = calendar.drop(['available', 'adjusted_price'], axis=1)
    
        # Provedení left join prvního datasetu s druhým datasetem
        merged_df = pd.merge(model_pop, additional_info, on='listing_id', how='left')
        merged_df = merged_df.groupby('listing_id').sample(n=1)
        
        return merged_df
    
    features = create_features(merged_df)

    # Uložení výsledného datasetu do CSV souboru
    features.to_csv(output_csv_path, index=False)
    
    print(f"Data byla úspěšně spojena a uložena do souboru {output_csv_path}")

merge_datasets_and_save("DataX---Banger-Team\\Data\\calendar.csv", "DataX---Banger-Team\\Data\\calendar.csv", "DataX---Banger-Team\\Data\\features_cal_cal.csv")
