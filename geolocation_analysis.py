import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from google.colab import drive

# Mount Google Drive (specific to Google Colab, adjust if running locally)
drive.mount('/content/drive')

# Load the datasets (Adjust the paths if running locally)
geo_data = pd.read_csv("/content/drive/MyDrive/covid19_tweets.csv")
cities = pd.read_csv("/content/drive/MyDrive/worldcities.csv")

# Preprocess the geolocation data
geo_data["location"] = geo_data["user_location"]
geo_data["country"] = np.NaN

# Split and clean user location to extract country information
user_location = geo_data['location'].fillna(value='').str.split(',')
world_city = cities['city'].fillna(value='').str.lower().str.strip().values.tolist()
world_states = cities['admin_name'].fillna(value='').str.lower().str.strip().tolist()
world_city_country = cities['country'].fillna(value='').str.lower().str.strip().values.tolist()
world_city_iso2 = cities['iso2'].fillna(value='').str.lower().str.strip().values.tolist()
world_city_iso3 = cities['iso3'].fillna(value='').str.lower().str.strip().values.tolist()

# Attempt to match user location with city, state, or country to assign a country
for i, loc_list in enumerate(user_location):
    for loc in loc_list:
        loc = loc.lower().strip()
        if loc in world_city:
            index = world_city.index(loc)
            geo_data.at[i, 'country'] = world_city_country[index]
            break
        elif loc in world_states:
            index = world_states.index(loc)
            geo_data.at[i, 'country'] = world_city_country[index]
            break
        elif loc in world_city_country:
            geo_data.at[i, 'country'] = loc
            break
        elif loc in world_city_iso2 or loc in world_city_iso3:
            geo_data.at[i, 'country'] = loc
            break

# Analyzing the geolocation data
# Plotting the number of tweets per country
plt.figure(figsize=(12, 8))
geo_data['country'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Countries by Number of Tweets')
plt.xlabel('Country')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)
plt.show()

# Animating the Spread over Time
def animate_spread(df, country_attribute, date_attribute, title="Spread over time"):
    fig = px.choropleth(df,
                        locations=country_attribute,
                        color=np.log(df["total_tweets"]+1), # Adding 1 to avoid log(0)
                        locationmode='country names',
                        animation_frame=date_attribute,
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title=title)
    fig.update_layout(coloraxis_colorbar=dict(title="Log(Tweet Count)"))
    fig.show()

# Prepare data for animation
geo_data['date'] = pd.to_datetime(geo_data['date']).dt.date
geo_data_grouped = geo_data.groupby(['date', 'country']).size().reset_index(name='total_tweets')

# Example: Animate the spread of tweets over time by country
animate_spread(geo_data_grouped, 'country', 'date')

# Note: This script assumes usage in Google Colab and specific file paths. Adjust paths and environment setup as necessary for your setup.
