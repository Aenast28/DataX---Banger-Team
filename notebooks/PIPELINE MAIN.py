import pandas as pd

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
         
        # ZDE MŮŽETE PROVÁDĚT CO CHCETE SE SVÝMI DATY
        # HLAVNĚ NECHTE NÁZEV VÝSLEDNÉHO DATAFRAME JAKO merged_df, ABY SE KOD NEMUSEL MĚNIT A PROBĚHL TAK, JAK JE NYNÍ

         return merged_df

    features = create_features(merged_df)

    # Uložení výsledného datasetu do CSV souboru
    features.to_csv(output_csv_path, index=False)
    
    print(f"Data byla úspěšně spojena a uložena do souboru {output_csv_path}")

merge_datasets_and_save("Data\\raw\\calendar.csv", "Data\\raw\\listings.csv", "Data\\raw\\features.csv")
