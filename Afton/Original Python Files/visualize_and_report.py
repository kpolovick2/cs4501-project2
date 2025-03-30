import folium
import pandas as pd
import random

# Load labeled cluster data
df_clusters = pd.read_csv("labeled_clusters.csv")

# Create a base map centered at the average latitude and longitude
map_center = [df_clusters["latitude"].mean(), df_clusters["longitude"].mean()]
map_ = folium.Map(location=map_center, zoom_start=12)

# Add markers for each cluster
for _, row in df_clusters.iterrows():
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=f"{row['place_type']} - {row['visits']} visits",
        icon=folium.Icon(color="blue" if row["visits"] > 5 else "green"),
    ).add_to(map_)

# Save map as HTML
map_.save("significant_locations_map.html")
print("Map created and saved as 'significant_locations_map.html'.")

# Load labeled clusters
df_clusters = pd.read_csv("labeled_clusters.csv")

# Sample a subset of labeled clusters for manual validation
sample_size = min(10, len(df_clusters))  # Avoid sampling error
if sample_size > 0:
    sample_clusters = df_clusters.sample(n=sample_size, random_state=42)
    print("Sample Locations for Manual Verification:")
    for _, row in sample_clusters.iterrows():
        print(f"Lat: {row['latitude']}, Lon: {row['longitude']}, Labeled: {row['place_type']}")
else:
    print("No clusters found to sample.")
