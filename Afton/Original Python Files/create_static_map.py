import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

# Load labeled cluster data
df_clusters = pd.read_csv("labeled_clusters.csv")

# Check if clusters are loaded correctly
if len(df_clusters) == 0:
    print("No clusters found to visualize. Exiting...")
    exit()

# Create GeoDataFrame for plotting
geometry = [Point(xy) for xy in zip(df_clusters["longitude"], df_clusters["latitude"])]
gdf_clusters = gpd.GeoDataFrame(df_clusters, geometry=geometry, crs="EPSG:4326")

# Convert to Web Mercator (EPSG:3857) for contextily compatibility
gdf_clusters = gdf_clusters.to_crs(epsg=3857)

# Set up a Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 8))

# Add a basemap (OpenStreetMap for now)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=14)

# Define colors for place types
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

# Set map bounds to fit all points
min_x, min_y, max_x, max_y = gdf_clusters.total_bounds
ax.set_xlim(min_x - 500, max_x + 500)
ax.set_ylim(min_y - 500, max_y + 500)

# Add title and labels
plt.title("Significant Locations and Clusters with Basemap")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Create a legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
if unique_labels:
    ax.legend(unique_labels.values(), unique_labels.keys(), title="Place Types")

# Save the map as PNG (optional)
plt.savefig("significant_locations_map_with_basemap.png", dpi=300)

# Show the map
plt.show()
