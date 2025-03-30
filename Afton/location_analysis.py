import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter
import requests
import time
from datetime import datetime, timedelta
import os
import folium

# Your Google API key
API_KEY = os.getenv("AIzaSyCoYdfEFsoHVp1fVNuitU6sQwhUg9ygDQc")

# STEP 1: Read Google Takeout Data
def load_location_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    one_month_ago = datetime(2025, 3, 20)

    coordinates = []
    for entry in data:
        if "startTime" in entry:
            entry_time = datetime.strptime(entry["startTime"][:19], "%Y-%m-%dT%H:%M:%S")
            if entry_time >= one_month_ago:
                continue

        if "visit" in entry and "topCandidate" in entry["visit"]:
            place_location = entry["visit"]["topCandidate"].get("placeLocation", "")
            if place_location:
                geo_data = place_location.split(":")[1]  
                lat, lon = map(float, geo_data.split(","))
                coordinates.append((lat, lon))
    return coordinates

# File path to your Google Takeout JSON
file_path = r'C:\Users\equus\CS4501\cs4501-project2\Afton\location-history.json'
coordinates = load_location_data(file_path)

# Convert to DataFrame
df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
print("LOCATION DATA \n")
print(df.head())

# STEP 2: Cluster GPS Coordinates
coords_array = np.radians(df[['latitude', 'longitude']].to_numpy())

# Apply DBSCAN clustering
db = DBSCAN(eps=0.00005, min_samples=5, metric='haversine').fit(coords_array)
print(f"Number of noise points: {(db.labels_ == -1).sum()}")

df['cluster'] = db.labels_

# STEP 3: Identify Significant Locations
def get_significant_locations(df):
    cluster_counts = Counter(df['cluster'])
    
    # Filter out noise (-1 label) and small clusters
    significant_clusters = {key: val for key, val in cluster_counts.items() if key != -1 and val > 5}
    
    # Get mean coordinates of significant clusters
    sig_locations = df[df['cluster'].isin(significant_clusters.keys())].groupby('cluster').agg({'latitude': 'mean', 'longitude': 'mean'}).reset_index()
    
    return sig_locations

significant_locations = get_significant_locations(df)
print("SIGNIFICANT LOCATIONS \n")
print(significant_locations)

# STEP 4: Label Places Using Google Places API
def get_place_info(lat, lon):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=50&key={API_KEY}"
    response = requests.get(url)
    data = response.json()

    try:
        if 'results' in data and len(data['results']) > 0:
            place = data['results'][0]
            name = place.get('name', 'Unknown')
            types = place.get('types', [])
            return name, types
        return None, None
    except requests.exceptions.ConnectionError:
        print("Error: Unable to connect to Google Maps API. Check your internet.")
    except requests.exceptions.Timeout:
        print("Error: Request to Google Maps API timed out.")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except Exception as e:
        print(f"Unexpected error: {e}")

significant_locations.loc[:, 'place_name'] = None
significant_locations.loc[:, 'place_type'] = None

for index, row in significant_locations.iterrows():
    name, types = get_place_info(row['latitude'], row['longitude'])
    significant_locations.at[index, 'place_name'] = name
    significant_locations.at[index, 'place_type'] = ', '.join(types) if types else None
    time.sleep(1)

print("SIGNIFICANT LOCATIONS w/ names \n")
print(significant_locations)

# STEP 5: Visualize Data on HTML Map
map_center = [df['latitude'].mean(), df['longitude'].mean()]
map_ = folium.Map(location=map_center, zoom_start=12)

# Add markers for significant locations
for _, row in significant_locations.iterrows():
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=f"{row['place_name']} ({row['place_type']})",
        icon=folium.Icon(color='blue')
    ).add_to(map_)

# Save map to HTML
map_.save("map.html")
print("Map created and saved as 'map.html'.")
