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

# Load world map from local file
world = gpd.read_file("ne_110m_admin_0_countries.shp")

# Set up a Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot a basic world map
world.plot(ax=ax, color="lightgray")

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
ax.set_xlim(min_x - 0.02, max_x + 0.02)  # Add padding to avoid cutting off points
ax.set_ylim(min_y - 0.02, max_y + 0.02)

# Add title and labels
plt.title("Significant Locations and Clusters")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Create a legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
if unique_labels:
    ax.legend(unique_labels.values(), unique_labels.keys(), title="Place Types")

# Save the map as a PNG (optional)
plt.savefig("significant_locations_map.png", dpi=300)

# Show the map with zoomed-in view
plt.show()
