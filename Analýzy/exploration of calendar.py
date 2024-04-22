# exploration of calendar.csv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
#pd.set_option('display.max_rows', 5000000)
#pd.set_option('display.max_columns', 5000000)

calendar = pd.read_csv("C:\\Users\\scott\\Downloads\\airbnb\\calendar.csv")

listings = len(calendar.listing_id.unique())
days = len(calendar.date.unique())

print(f'The are {listings} unique listings over {days} days.')

print(f'The listings start on {calendar.date.min()} and end {calendar.date.max()}')

calendar.head()

calendar.info(verbose=True, show_counts=True) #all non null except adj.price
calendar = calendar.drop('adjusted_price', axis=1)


# availability throughout the year
calendar.available.value_counts()

calendar_new = calendar[['date', 'available']]
calendar_new['busy'] = calendar_new.available.map( lambda x: 0 if x == 't' else 1)

calendar_new.head()

calendar_new = calendar_new.groupby('date')['busy'].mean().reset_index()
calendar_new['date'] = pd.to_datetime(calendar_new['date'])

calendar_new.head()

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.figure(figsize=(10, 5))
plt.plot(calendar_new['date'], calendar_new['busy'])
plt.title('Airbnb Prague Calendar')
plt.ylabel('% busy')
plt.show()

calendar['date'] = pd.to_datetime(calendar['date'])

def get_cleaned_price(price: pd.core.series.Series) -> float:
    """ Returns a float price from a pandas Series including the currency """
    return price.str.replace('$', '').str.replace(',', '').astype(float)

calendar['price'] = get_cleaned_price(calendar['price'])

calendar.head()

mean_per_month = calendar.groupby(calendar['date'].dt.strftime('%B'), sort=False)['price'].mean()

mean_per_month.plot(kind = 'barh' , figsize = (12,7))
plt.xlabel('average monthly price')

calendar['day_of_the_week'] = calendar.date.dt.day_name()
calendar.head()

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
price_week = calendar[['day_of_the_week', 'price']]
price_week.head()

price_week = price_week.groupby(['day_of_the_week']).mean().reindex(days)
price_week

price_week.plot()
ticks = list(range(0, 7, 1))
labels = "Mon Tues Weds Thurs Fri Sat Sun".split()
plt.xticks(ticks, labels)

calendar['price'].value_counts()
calendar['price'].mean()

mean_prices = calendar.groupby('date')['price'].mean()

# Plotting the mean prices over time
plt.figure(figsize=(10, 5))
plt.plot(mean_prices.index, mean_prices.values, color='b')
plt.title('Mean Price by Date')
plt.xlabel('Date')
plt.ylabel('Mean Price')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()