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

# Your Google API key (not used in the new method, but kept for legacy purposes)
API_KEY = os.getenv("AIzaSyCNyMpVn9XKflzM-vPVTWsp2w_oI0e__lQ")

# STEP 1: Read Google Takeout Data
def load_location_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    start_date = datetime(2025, 1, 20)  # Start date: 1/20/2025
    end_date = datetime(2025, 3, 16)    # End date: 3/16/2025

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
    cluster_counts = cluster_data.groupby(['latitude', 'longitude']).size().reset_index(name='count')
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

# STEP 5: Label Places Using OpenStreetMap's Nominatim API (to get physical address)
def get_osm_address(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "json",
        "lat": lat,
        "lon": lon,
        "addressdetails": 1
    }
    headers = {'User-Agent': 'Mozilla/5.0 (YourAppName)'}
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            print(f"Error: Received HTTP status code {response.status_code} for ({lat}, {lon})")
            return "Unknown"
        data = response.json()
        # Debug: Print full response if needed
        print(f"OSM API response for ({lat}, {lon}): {json.dumps(data, indent=2)}")
        if "error" in data:
            print(f"Error in response for ({lat}, {lon}): {data['error']}")
            return "Unknown"
        return data.get("display_name", "Unknown")
    except Exception as e:
        print(f"Unexpected error for ({lat}, {lon}): {e}")
        return "Unknown"

significant_locations['address'] = None

# Query the Nominatim API for each significant location to get the address
for index, row in significant_locations.iterrows():
    address = get_osm_address(row['latitude'], row['longitude'])
    significant_locations.at[index, 'address'] = address
    time.sleep(1)  # Respect Nominatim's usage policy

# Check the first few rows to verify the addresses
print(significant_locations.head())

# STEP 6: Save Summary Information to CSV (only coordinates, visit_count, and address)
summary_data = []
for index, row in significant_locations.iterrows():
    summary_data.append({
        'Latitude': row['latitude'],
        'Longitude': row['longitude'],
        'Visit Count': row['visit_count'],
        'Address': row['address']
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("visit_summary.csv", index=False)
print(f"Summary data saved to 'visit_summary.csv'.")

# STEP 7: Create Map Visualization
def create_map(df, significant_locations):
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
    for index, row in significant_locations.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Location: {row['address']}<br>Visit Count: {row['visit_count']}",
            icon=folium.Icon(color='blue')
        ).add_to(m)
    m.save("map.html")
    print("Map saved to 'map.html'.")

create_map(df, significant_locations)

# STEP 8: Write a formatted text summary
# Format as requested:
# Significant Clusters of Locations:  1/20/2025 - 3/16/2025
# Afton Mueller
# Coordinates:  ____, ____
# Visit Count:  ____
# Address:  ____

# Since "Location Type" is not determined in this code, we'll use a placeholder "N/A".

with open("visit_summary.txt", "w") as f:
    f.write("Significant Clusters of Locations: 1/20/2025 - 3/16/2025\n")
    f.write("Afton Mueller\n\n")
    for _, row in summary_df.iterrows():
        f.write(f"Coordinates: {row['Latitude']}, {row['Longitude']}\n")
        f.write(f"Visit Count: {row['Visit Count']}\n")
        f.write(f"Address: {row['Address']}\n")

print("Formatted summary saved to 'visit_summary.txt'.")
