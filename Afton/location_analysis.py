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
API_KEY = os.getenv("API Key")

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

# NEW STEP: Add function to determine location type based on address
def determine_location_type(address, amenity=None):
    """
    Determine the type of location based on keywords in the address or amenity information.
    
    Args:
        address (str): The address string from OSM
        amenity (str, optional): Specific amenity type if available from OSM
        
    Returns:
        str: The determined location type
    """
    # Define location types and their associated keywords
    location_types = {
        'Residential': ['house', 'apartment', 'residential', 'home', 'flat', 'condo', 'housing'],
        'Educational': ['university', 'college', 'school', 'campus', 'library', 'academy', 'institute'],
        'Commercial': ['shop', 'store', 'mall', 'market', 'supermarket', 'retail', 'shopping'],
        'Food & Dining': ['restaurant', 'café', 'cafe', 'diner', 'pizzeria', 'bar', 'pub', 'eatery', 'bakery', 'coffee'],
        'Office': ['office', 'building', 'corporate', 'business', 'headquarters', 'workplace', 'suite'],
        'Healthcare': ['hospital', 'clinic', 'medical', 'healthcare', 'doctor', 'pharmacy', 'dentist', 'health'],
        'Recreation': ['park', 'garden', 'playground', 'recreation', 'gym', 'fitness', 'stadium', 'theater', 'cinema'],
        'Transportation': ['station', 'airport', 'bus', 'train', 'subway', 'transit', 'terminal', 'stop'],
        'Lodging': ['hotel', 'motel', 'inn', 'hostel', 'lodge', 'resort', 'accommodation'],
        'Religious': ['church', 'temple', 'mosque', 'synagogue', 'chapel', 'religious', 'worship']
    }
    
    # If amenity is provided and non-empty, check it first
    if amenity and amenity.strip():
        lower_amenity = amenity.lower()
        for loc_type, keywords in location_types.items():
            if any(keyword in lower_amenity for keyword in keywords):
                return loc_type
    
    # Check address
    if address and address.strip():
        lower_address = address.lower()
        for loc_type, keywords in location_types.items():
            if any(keyword in lower_address for keyword in keywords):
                return loc_type
    
    # Default return value
    return 'Unknown'

# Modify the get_osm_address function to also retrieve amenity information
def get_osm_data(lat, lon):
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
            return "Unknown", None
        data = response.json()
        # Extract address and amenity information
        address = data.get("display_name", "Unknown")
        
        # Try to extract amenity information from different possible fields
        amenity = None
        if "type" in data:
            amenity = data["type"]
        elif "category" in data:
            amenity = data["category"]
        elif "amenity" in data.get("address", {}):
            amenity = data["address"]["amenity"]
        
        return address, amenity
    except Exception as e:
        print(f"Unexpected error for ({lat}, {lon}): {e}")
        return "Unknown", None

# Initialize columns
significant_locations['address'] = None
significant_locations['location_type'] = None

# Query the Nominatim API for each significant location to get the address and determine location type
for index, row in significant_locations.iterrows():
    address, amenity = get_osm_data(row['latitude'], row['longitude'])
    significant_locations.at[index, 'address'] = address
    
    # Determine location type based on address and amenity
    location_type = determine_location_type(address, amenity)
    significant_locations.at[index, 'location_type'] = location_type
    
    time.sleep(1)  # Respect Nominatim's usage policy

# Check the first few rows to verify the addresses and location types
print(significant_locations.head())

# STEP 6: Save Summary Information to CSV (including location_type)
summary_data = []
for index, row in significant_locations.iterrows():
    summary_data.append({
        'Latitude': row['latitude'],
        'Longitude': row['longitude'],
        'Visit Count': row['visit_count'],
        'Address': row['address'],
        'Location Type': row['location_type']
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("visit_summary.csv", index=False)
print(f"Summary data saved to 'visit_summary.csv'.")

# STEP 7: Create Enhanced Map Visualization with location types
def create_map(df, significant_locations):
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
    
    # Define colors for different location types
    type_colors = {
        'Residential': 'blue',
        'Educational': 'green',
        'Commercial': 'red',
        'Food & Dining': 'orange',
        'Office': 'purple',
        'Healthcare': 'pink',
        'Recreation': 'darkgreen',
        'Transportation': 'gray',
        'Lodging': 'cadetblue',
        'Religious': 'darkpurple',
        'Unknown': 'black'
    }
    
    for index, row in significant_locations.iterrows():
        # Get color based on location type
        color = type_colors.get(row['location_type'], 'black')
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Location: {row['address']}<br>Visit Count: {row['visit_count']}<br>Type: {row['location_type']}",
            icon=folium.Icon(color=color)
        ).add_to(m)
        
    # Add a legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
    <h4>Location Types</h4>
    '''
    
    for loc_type, color in type_colors.items():
        legend_html += f'<i style="background: {color}; width: 15px; height: 15px; display: inline-block;"></i> {loc_type}<br>'
    
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save("location_map.html")
    print("Enhanced map with location types saved to 'location_map.html'.")

create_map(df, significant_locations)

# STEP 8: Write a formatted text summary including location types
with open("visit_summary.txt", "w") as f:
    f.write("Significant Clusters of Locations: 1/20/2025 - 3/16/2025\n")
    f.write("Afton Mueller\n\n")
    for _, row in summary_df.iterrows():
        f.write(f"Coordinates: {row['Latitude']}, {row['Longitude']}\n")
        f.write(f"Visit Count: {row['Visit Count']}\n")
        f.write(f"Location Type: {row['Location Type']}\n")
        f.write(f"Address: {row['Address']}\n\n")

print("Formatted summary with location types saved to 'visit_summary.txt'.")

# STEP 9: Generate Location Type Statistics
location_type_counts = significant_locations['location_type'].value_counts()
print("\nLocation Type Distribution:")
print(location_type_counts)

# Add the statistics to the text summary
with open("visit_summary.txt", "a") as f:
    f.write("\n\nLocation Type Distribution:\n")
    for loc_type, count in location_type_counts.items():
        f.write(f"{loc_type}: {count} locations\n")

print("Location type statistics added to 'visit_summary.txt'.")