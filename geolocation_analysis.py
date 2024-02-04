import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load the geolocation dataset
geo_data = pd.read_csv("/content/drive/MyDrive/covid19_tweets.csv")

# Display the first few rows to understand the dataset's structure
print(geo_data.head())

# Get the dataset's dimensions
rows, cols = geo_data.shape
print(f"There are {rows} rows and {cols} columns.")

# Display general info about the dataset
geo_data.info()

# Plotting the number of non-null values for each column
non_null_counts = geo_data.notna().sum()
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.barplot(x=non_null_counts.index, y=non_null_counts.values, palette="viridis")
plt.xticks(rotation=90)
plt.xlabel("Columns")
plt.ylabel("Count of Non-Null Values")
plt.title("Count of Non-Null Values in Each Column")
plt.tight_layout()
plt.show()

# Dropping unnecessary columns for geolocation analysis
columns_to_drop = ['user_description', 'user_followers', 'user_friends', 'user_favourites', 'user_verified', 'source', 'is_retweet', 'hashtags', 'user_created']
geo_data = geo_data.drop(columns=columns_to_drop)

# Splitting date and time, keeping only the date
geo_data['date'] = pd.to_datetime(geo_data['date'])
geo_data = geo_data.sort_values(['date'])
geo_data['date'] = geo_data['date'].astype(str).str.split(' ', expand=True)[0]

# Function to display the range of dates in the dataset
def date_range(df):
    min_date = df['date'].min()
    max_date = df['date'].max()
    print(f"Date Range: From {min_date} to {max_date}")

date_range(geo_data)

# Function to plot the locations with the most tweets
def locations_with_most_tweets(df, loc_attribute):
    ds = df[loc_attribute].value_counts().reset_index()
    ds.columns = [loc_attribute, 'Tweet Count']
    ds = ds[ds[loc_attribute] != 'NA']
    fig = px.bar(
        x="Tweet Count",
        y=loc_attribute,
        orientation='h', 
        title='Top 10 user locations by number of tweets',
        color=loc_attribute,
        width=800,
        height=800,
        data_frame=ds[:15]
    )
    fig.show()

# Plotting the user locations with the most tweets
locations_with_most_tweets(geo_data, 'user_location')

# Feature engineering the location attribute to get the name of the country
cities = pd.read_csv("/content/drive/MyDrive/worldcities.csv")
geo_data["location"] = geo_data["user_location"]
geo_data["country"] = np.NaN
user_location = geo_data['location'].fillna(value='').str.split(',')

# Lists to hold city and country data
lat = cities['lat'].fillna(value='').values.tolist()
lng = cities['lng'].fillna(value='').values.tolist()
country = cities['country'].fillna(value='').values.tolist()

# Populate lists with unique identifiers
world_city_iso3, world_city_iso2, world_city_country, world_states, world_city = [], [], [], [], []
for c in cities['iso3'].str.lower().str.strip().values.tolist():
    if c not in world_city_iso3:
        world_city_iso3.append(c)
for c in cities['iso2'].str.lower().str.strip().values.tolist():
    if c not in world_city_iso2:
        world_city_iso2.append(c)
for c in cities['country'].str.lower().str.strip().values.tolist():
    if c not in world_city_country:
        world_city_country.append(c)
for c in cities['admin_name'].str.lower().str.strip().tolist():
    world_states.append(c)
world_city = cities['city'].fillna(value='').str.lower().str.strip().values.tolist()

# Assign country based on the location data
for each_loc in range(len(user_location)):
    ind = each_loc
    each_loc = user_location[each_loc]
    for each in each_loc:
        each = each.lower().strip()
        if each in world_city:
            order = world_city.index(each)
            geo_data['country'][ind] = country[order].lower()
            continue
        if each in world_states:
            order = world_states.index(each)
            geo_data['country'][ind] = country[order].lower()
            continue
        if each in world_city_country:
            order = world_city_country.index(each)
            geo_data['country'][ind] = world_city_country[order].lower()
            continue
        if each in world_city_iso2:
            order = world_city_iso2.index(each)
            geo_data['country'][ind] = world_city_country[order].lower()
            continue
        if each in world_city_iso3:
            order = world_city_iso3.index(each)
            geo_data['country'][ind] = world_city_country[order].lower()
            continue

# Plotting the actual countries with the most tweets after feature engineering
locations_with_most_tweets(geo_data, 'country')

# Animating the spread over time
def animate_spread(df, country_attribute, day_attribute, spread_type):
    fig = px.choropleth(
        df,
        locations=country_attribute,
        color=spread_type,
        locationmode='country names',
        animation_frame=day_attribute,
        title='Spread over time'
    )
    fig.show()

# Prepare data for animating the spread of tweets over time
geo_data_cummulative = geo_data.copy()
geo_data_cummulative = geo_data_cummulative.drop(columns=['user_name', 'user_location', 'location'])
geo_data_cummulative = geo_data_cummulative.dropna(subset=['country'])
geo_data_cummulative = geo_data_cummulative.sort_values(by=['date'])
geo_data_cummulative['total_tweets'] = geo_data_cummulative.groupby('country').cumcount() + 1
geo_data_cummulative.reset_index(drop=True, inplace=True)

# Create a DataFrame of all possible combinations of countries and dates
all_countries = geo_data_cummulative['country'].unique()
all_dates = geo_data_cummulative['date'].unique()
multi_index = pd.MultiIndex.from_product([all_countries, all_dates], names=['country', 'date'])
all_combinations_df = pd.DataFrame(index=multi_index).reset_index()

# Merge with the original DataFrame and forward fill the cumulative tweets for each country
merged_total_tweets = pd.merge(all_combinations_df, geo_data_cummulative, on=['country', 'date'], how='left')
merged_total_tweets['total_tweets'] = merged_total_tweets.groupby('country')['total_tweets'].ffill().fillna(0)

# Animate the spread of tweets over time
animate_spread(merged_total_tweets, 'country', 'date', 'total_tweets')
