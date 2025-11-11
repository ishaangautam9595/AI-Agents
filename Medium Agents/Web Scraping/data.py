import streamlit as st
import requests
import time

# Replace with your Google Places API key
API_KEY = "API_KEY"

def get_places_info(api_key, location, radius=15000, query="school"):
    """
    Fetches places near a given location using the Google Places API.
    Handles pagination to retrieve more than 20 results.
    """
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": location,
        "radius": radius,
        "keyword": query,
        "key": api_key
    }
    
    places_info = []
    next_page_token = None
    
    while True:
        # Add the next_page_token to the params if it exists
        if next_page_token:
            params["pagetoken"] = next_page_token
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check if the response contains results
        if "results" in data:
            results = data["results"]
            for result in results:
                place_id = result.get("place_id")
                place_details = get_place_details(api_key, place_id)
                
                if place_details:
                    places_info.append(place_details)
        
        # Check if there is a next page
        next_page_token = data.get("next_page_token")
        if not next_page_token:
            break  # Exit the loop if there are no more pages
        
        # Wait for a short time before making the next request (required by Google API)
        time.sleep(2)
    
    return places_info

def get_place_details(api_key, place_id):
    """
    Fetches detailed information about a place using its place_id, including latitude, longitude, and rating.
    """
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,formatted_address,geometry/location,rating",
        "key": api_key
    }
    
    response = requests.get(url, params=params)
    result = response.json().get("result", {})
    
    place_details = {
        "name": result.get("name"),
        "address": result.get("formatted_address", "N/A"),
        "latitude": result.get("geometry", {}).get("location", {}).get("lat", "N/A"),
        "longitude": result.get("geometry", {}).get("location", {}).get("lng", "N/A"),
        "rating": result.get("rating", "N/A")
    }
    
    return place_details

def geocode_location(api_key, location_name):
    """
    Converts a location name (e.g., "Ambala, Haryana, India") into latitude and longitude.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": location_name,
        "key": api_key
    }
    
    response = requests.get(url, params=params)
    results = response.json().get("results", [])
    
    if results:
        location = results[0]["geometry"]["location"]
        return f"{location['lat']},{location['lng']}"
    else:
        return None

def main():
    st.title("Google Places Data Scraper")
    st.write("Enter a location and search for places (e.g., schools) using the Google Places API.")
    
    # Input fields
    location_input = st.text_input("Enter a location (e.g., 'Ambala, Haryana, India' or '30.3782,76.7767'):")
    query = st.text_input("Enter the type of place to search for (e.g., 'school'):", "school")
    radius = st.number_input("Enter the search radius in meters:", min_value=100, value=5000)
    
    if st.button("Search"):
        if location_input:
            # Check if the input is already in latitude,longitude format
            if "," in location_input and all(part.strip().replace(".", "").replace("-", "").isdigit() for part in location_input.split(",")):
                location = location_input
            else:
                # Geocode the location name to get latitude and longitude
                location = geocode_location(API_KEY, location_input)
            
            if location:
                st.write(f"Searching for '{query}' near {location_input}...")
                places_info = get_places_info(API_KEY, location, radius, query)
                
                if places_info:
                    st.write(f"Found {len(places_info)} places:")
                    for place in places_info:
                        st.write(f"**Name:** {place['name']}")
                        st.write(f"**Address:** {place['address']}")
                        st.write(f"**Latitude:** {place['latitude']}")
                        st.write(f"**Longitude:** {place['longitude']}")
                        st.write(f"**Rating:** {place['rating']} / 5")
                        st.write("---")
                else:
                    st.warning("No places found.")
            else:
                # If geocoding fails, try a default location (e.g., Ambala, Haryana, India)
                st.warning(f"Could not geocode '{location_input}'. Using default location: Ambala, Haryana, India.")
                default_location = "30.3782,76.7767"  # Coordinates for Ambala, Haryana, India
                st.write(f"Searching for '{query}' near Ambala, Haryana, India...")
                places_info = get_places_info(API_KEY, default_location, radius, query)
                
                if places_info:
                    st.write(f"Found {len(places_info)} places:")
                    for place in places_info:
                        st.write(f"**Name:** {place['name']}")
                        st.write(f"**Address:** {place['address']}")
                        st.write(f"**Latitude:** {place['latitude']}")
                        st.write(f"**Longitude:** {place['longitude']}")
                        st.write(f"**Rating:** {place['rating']} / 5")
                        st.write("---")
                else:
                    st.warning("No places found.")
        else:
            st.warning("Please enter a location.")

if __name__ == "__main__":
    main()