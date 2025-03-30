import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from utils import get_place_type
import time

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

# Query Google Places API to label cluster centers
def label_cluster(row):
    """Label the cluster center with a place type using Google Places API."""
    return get_place_type(row["latitude"], row["longitude"])

clustered_locations["place_type"] = clustered_locations.apply(label_cluster, axis=1)

# Save labeled clusters
clustered_locations.to_csv("labeled_clusters.csv", index=False)
print(f"Labeled clusters saved successfully with {len(clustered_locations)} entries.")
