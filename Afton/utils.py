import requests
import time

# Google Places API Key
API_KEY = "api key"  
BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

def get_place_type(lat, lon, radius=50):
    """Get the place type using Google Places API."""
    params = {
        "location": f"{lat},{lon}",
        "radius": radius,
        "key": API_KEY,
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200 and response.json().get("results"):
        return response.json()["results"][0]["types"][0]
    return "Unknown"
