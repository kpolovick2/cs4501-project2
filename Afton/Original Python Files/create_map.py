import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Load labeled cluster data
df_clusters = pd.read_csv("labeled_clusters.csv")

# Check if clusters are loaded correctly
if len(df_clusters) == 0:
    print("No clusters found to visualize. Exiting...")
    exit()

# Create GeoDataFrame for plotting
geometry = [Point(xy) for xy in zip(df_clusters["longitude"], df_clusters["latitude"])]
gdf_clusters = gpd.GeoDataFrame(df_clusters, geometry=geometry)

# Set up a Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot a basic world map using GeoPandas
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world.plot(ax=ax, color="lightgray")

# Plot cluster points
colors = {"home": "red", "work": "blue", "restaurant": "orange", "school": "green", "other": "purple"}

# Plot all clusters with color-coded markers
for _, row in gdf_clusters.iterrows():
    ax.scatter(
        row.geometry.x,
        row.geometry.y,
        s=row["visits"] * 5,  # Scale marker size by visit count
        c=colors.get(row["place_type"].lower(), "purple"),
        label=row["place_type"] if row["place_type"] not in colors else "",
        alpha=0.7,
        edgecolor="k",
    )

# Add title and labels
plt.title("Significant Locations and Clusters")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Create a legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
if unique_labels:
    ax.legend(unique_labels.values(), unique_labels.keys(), title="Place Types")

# Show the map
plt.show()
