import json
import pandas as pd
from datetime import datetime

# Corrected file path
file_path = r"C:\Users\equus\CS4501\cs4501-project2\Afton\location-history.json"

# Load JSON data
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Check if the data is a list
if not isinstance(data, list):
    raise TypeError("Data is not in the expected list format. Check your JSON file.")

# Create empty lists for visits and activities
visits_data = []
activities_data = []

# Process each entry in the data
for entry in data:
    if "visit" in entry:
        visit = entry["visit"]["topCandidate"]
        visits_data.append({
            "type": "visit",
            "latitude": float(visit["placeLocation"].split(":")[1].split(",")[0]),
            "longitude": float(visit["placeLocation"].split(",")[1]),
            "start_time": datetime.fromisoformat(entry["startTime"]),
            "end_time": datetime.fromisoformat(entry["endTime"]),
            "duration_hours": (datetime.fromisoformat(entry["endTime"]) - datetime.fromisoformat(entry["startTime"])).total_seconds() / 3600,
            "place_id": visit["placeID"],
            "semantic_type": visit["semanticType"],
        })
    
    elif "activity" in entry:
        activity = entry["activity"]["topCandidate"]
        activities_data.append({
            "type": "activity",
            "latitude_start": float(entry["activity"]["start"].split(":")[1].split(",")[0]),
            "longitude_start": float(entry["activity"]["start"].split(",")[1]),
            "latitude_end": float(entry["activity"]["end"].split(":")[1].split(",")[0]),
            "longitude_end": float(entry["activity"]["end"].split(",")[1]),
            "start_time": datetime.fromisoformat(entry["startTime"]),
            "end_time": datetime.fromisoformat(entry["endTime"]),
            "duration_hours": (datetime.fromisoformat(entry["endTime"]) - datetime.fromisoformat(entry["startTime"])).total_seconds() / 3600,
            "activity_type": activity["type"],
        })

# Convert visits and activities to DataFrames
df_visits = pd.DataFrame(visits_data)
df_activities = pd.DataFrame(activities_data)

# Save data to CSV
df_visits.to_csv("visits_data.csv", index=False)
df_activities.to_csv("activities_data.csv", index=False)

print(f"Visits data saved with {len(df_visits)} entries.")
print(f"Activities data saved with {len(df_activities)} entries.")
