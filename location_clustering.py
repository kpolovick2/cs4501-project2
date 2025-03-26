import pandas as pd
from sklearn.cluster import DBSCAN
import json
import matplotlib.pyplot as plt


# Function to read the JSON file and extract coordinates
def extract_coordinates_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    coordinates = []
    for entry in data:
        # Extract latitude and longitude from the 'placeLocation' field
        if "visit" in entry and "topCandidate" in entry["visit"]:
            place_location = entry["visit"]["topCandidate"].get("placeLocation", "")
            if place_location:
                geo_data = place_location.split(":")[1]  # Extract after 'geo:'
                lat, lon = map(float, geo_data.split(","))
                coordinates.append((lat, lon))
    return coordinates


# Step 1: Parse the JSON data and extract the GPS coordinates
file_path = '/Users/kearapolovick/Desktop/location-history.json'
coordinates = extract_coordinates_from_file(file_path)

# Check if coordinates are extracted
if not coordinates:
    print("No valid coordinates found in the file.")
else:
    # Convert the coordinates to a pandas DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Step 2: DBSCAN clustering on the coordinates
    # Convert latitudes and longitudes to radians for geodesic distance computation
    coords_in_radians = df.apply(lambda x: [x['latitude'], x['longitude']], axis=1).tolist()

    # DBSCAN clustering
    db = DBSCAN(eps=0.0005, min_samples=2, metric='haversine').fit(coords_in_radians)

    # Add the cluster labels to the DataFrame
    df['cluster'] = db.labels_

    # Print the DataFrame with the cluster labels
    print(df)

    # The clusters for the coordinates
    print("Clustered Coordinates:")
    print(df.groupby('cluster').agg({'latitude': 'mean', 'longitude': 'mean'}))

    # Visualization of the clusters
    plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], cmap='viridis')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('DBSCAN Clusters of Locations')
    plt.colorbar(label='Cluster ID')
    plt.show()

