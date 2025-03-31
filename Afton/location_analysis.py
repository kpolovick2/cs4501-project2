import json
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from collections import Counter
from sklearn.cluster import DBSCAN
import os
from sklearn.metrics import pairwise_distances_argmin_min
import folium  # For map visualization

# Your Google API key
API_KEY = os.getenv("AIzaSyCoYdfEFsoHVp1fVNuitU6sQwhUg9ygDQc")

# STEP 1: Read Google Takeout Data
def load_location_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    start_date = datetime(2025, 1, 20)  # Start date: 1/20/2025
    end_date = datetime(2025, 3, 16)   # End date: 3/16/2025

    coordinates = []
    for entry in data:
        if "startTime" in entry:
            entry_time = datetime.strptime(entry["startTime"][:19], "%Y-%m-%dT%H:%M:%S")
            
            # Include data only within the specified date range
            if entry_time < start_date or entry_time > end_date:
                continue

        if "visit" in entry and "topCandidate" in entry["visit"]:
            place_location = entry["visit"]["topCandidate"].get("placeLocation", "")
            if place_location:
                geo_data = place_location.split(":")[1]  
                lat, lon = map(float, geo_data.split(","))
                coordinates.append((lat, lon, entry_time))  # include timestamp for time-based analysis
    return coordinates

# File path to your Google Takeout JSON
file_path = r'C:\Users\equus\CS4501\cs4501-project2\Afton\location-history.json'
coordinates = load_location_data(file_path)

# Convert to DataFrame
df = pd.DataFrame(coordinates, columns=['latitude', 'longitude', 'timestamp'])

# STEP 2: Cluster GPS Coordinates
# Convert coordinates to radians for DBSCAN clustering
coords_array = np.radians(df[['latitude', 'longitude']].to_numpy())

# Apply DBSCAN clustering for spatial clustering
db = DBSCAN(eps=0.00005, min_samples=5, metric='haversine').fit(coords_array)

df['cluster'] = db.labels_

# STEP 3: Filter Clusters and Determine Frequent Locations
def get_frequent_location_from_cluster(cluster_data):
    """
    Given a cluster of points, identify the most frequently visited location in that cluster.
    """
    # Calculate the most frequent coordinates in this cluster
    cluster_counts = cluster_data.groupby(['latitude', 'longitude']).size().reset_index(name='count')
    # Sort by count to pick the most visited location
    most_visited = cluster_counts.sort_values(by='count', ascending=False).iloc[0]
    
    return most_visited['latitude'], most_visited['longitude'], most_visited['count']

# STEP 4: Identify Significant Locations
def get_significant_locations(df):
    cluster_counts = Counter(df['cluster'])
    
    # Filter out noise (-1 label) and small clusters, only keep clusters with occurrences of 6 or greater
    significant_clusters = {key: val for key, val in cluster_counts.items() if key != -1 and val >= 6}
    
    significant_locations = []
    
    for cluster in significant_clusters.keys():
        cluster_data = df[df['cluster'] == cluster]
        lat, lon, visit_count = get_frequent_location_from_cluster(cluster_data)
        significant_locations.append({
            'latitude': lat,
            'longitude': lon,
            'visit_count': visit_count,
            'cluster': cluster
        })
    
    return pd.DataFrame(significant_locations)

significant_locations = get_significant_locations(df)

# STEP 5: Label Places Using Google Places API
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

significant_locations['place_name'] = None
significant_locations['place_type'] = None

# Query the Google Places API for each significant location
for index, row in significant_locations.iterrows():
    name, types = get_place_info(row['latitude'], row['longitude'])
    significant_locations.at[index, 'place_name'] = name
    significant_locations.at[index, 'place_type'] = ', '.join(types) if types else None
    time.sleep(1)

# Check the first few rows to debug the "Unknown" labels
print(significant_locations.head())

# STEP 6: Save Summary Information to CSV
summary_data = []

for index, row in significant_locations.iterrows():
    location = row['place_name'] if row['place_name'] else "Unknown"
    place_type = row['place_type'] if row['place_type'] else "Unknown"
    visit_count = row['visit_count']
    summary_data.append({
        'Location Name': location,
        'Place Type': place_type,
        'Latitude': row['latitude'],
        'Longitude': row['longitude'],
        'Visit Count': visit_count
    })

# Save to CSV for further analysis
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("visit_summary.csv", index=False)
print(f"Summary data saved to 'visit_summary.csv'.")

# STEP 7: Create Map Visualization
def create_map(df, significant_locations):
    # Create a base map
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)

    # Add markers for significant locations
    for index, row in significant_locations.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Location: {row['place_name']}<br>Visit Count: {row['visit_count']}<br>Place Type: {row['place_type']}",
            icon=folium.Icon(color='blue')
        ).add_to(m)
    
    # Save to an HTML file
    m.save("map.html")
    print("Map saved to 'map.html'.")

# Generate the map
create_map(df, significant_locations)
