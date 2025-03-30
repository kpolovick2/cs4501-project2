import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import googlemaps
import time

# Initialize the Google Maps client with your API key
gmaps = googlemaps.Client(key="AIzaSyCoYdfEFsoHVp1fVNuitU6sQwhUg9ygDQc")

# Load cleaned visits data
df_visits = pd.read_csv("cleaned_visits_data.csv")

# DBSCAN Parameters
coords = df_visits[["latitude", "longitude"]].values
epsilon = 50 / 111_320  # 50 meters in degrees of latitude/longitude
min_samples = 5

# Apply DBSCAN to cluster nearby points
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm="ball_tree", metric="haversine").fit(np.radians(coords))
df_visits["cluster"] = dbscan.labels_

# Get cluster centers and count visits per cluster
clustered_locations = df_visits.groupby("cluster").agg(
    latitude=("latitude", "mean"),
    longitude=("longitude", "mean"),
    visits=("cluster", "count"),
).reset_index()

# Remove noise points (-1 cluster)
clustered_locations = clustered_locations[clustered_locations["cluster"] != -1]

# Function to reverse geocode and fetch business name
def reverse_geocode(lat, lng):
    """Fetches the address and business name for the given coordinates."""
    try:
        result = gmaps.reverse_geocode((lat, lng))
        if result:
            address = result[0].get('formatted_address')
            business_name = None
            for component in result[0].get('address_components', []):
                if 'point_of_interest' in component.get('types', []):
                    business_name = component.get('long_name')
                    break
            return address, business_name
    except Exception as e:
        print(f"Error fetching data for ({lat}, {lng}): {e}")
    return None, None

# Function to classify place type
def classify_location(address, business_name):
    """Classifies the location as Home, Work, School, Business, or Other."""
    if not address:
        return "Other"

    address_lower = address.lower()
    business_lower = business_name.lower() if business_name else ""

    # Check for school/university
    school_keywords = ["school", "university", "college", "academy", "institute"]
    if any(word in address_lower or word in business_lower for word in school_keywords):
        return "School"

    # Check for business
    business_keywords = ["restaurant", "store", "shop", "cafe", "hotel", "mall", "company", "corporation"]
    if any(word in address_lower or word in business_lower for word in business_keywords):
        return "Business"

    # Check for work-related locations
    work_keywords = ["office", "corporate", "headquarters", "firm"]
    if any(word in address_lower or word in business_lower for word in work_keywords):
        return "Work"

    # Assume frequent late-night location is home (if data available)
    # This would require a separate check of timestamps for late-night presence
    # For now, we default to "Other" if unknown
    return "Other"

# Apply reverse geocoding and classification
addresses = []
business_names = []
location_types = []

for index, row in clustered_locations.iterrows():
    address, business_name = reverse_geocode(row['latitude'], row['longitude'])
    addresses.append(address)
    business_names.append(business_name)
    location_types.append(classify_location(address, business_name))
    time.sleep(0.5)  # Sleep to avoid hitting API rate limits

clustered_locations['address'] = addresses
clustered_locations['business_name'] = business_names
clustered_locations['place_type'] = location_types

# Save labeled clusters
clustered_locations.to_csv("enhanced_clusters_labeled.csv", index=False)
print(f"Enhanced clusters saved successfully with {len(clustered_locations)} entries.")
