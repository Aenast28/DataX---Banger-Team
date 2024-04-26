import pandas as pd

def features_job(data):
    # Načtení souboru
    data = pd.read_csv(r"Data\\listings.csv")  # Použití surového řetězce pro cestu
    # Úpravy sloupců
    data = data.iloc[:25, :]
    data.rename(columns={'id': 'listing_id'}, inplace=True)

    # Načtení modelovací populace
    mode_pop = pd.read_csv(r"DataX---Banger-Team\\Data\\calendar.csv")  # Opravená cesta
    merged = pd.merge(mode_pop, data, on='listing_id', how='left')  # Opravené volání merge

    # Krok 2 - create features
    def create_features(merged):
        # Všechny relevantní úpravy, tvoření nových sloupců, diskretizace apod.
        return merged

    features = create_features(merged)

    # Krok 3 - save features
    features.to_csv(r"DataX---Banger-Team\\Data\\features.csv", index=False)  # Opravené uložení
