from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import requests
import logging
import json

logger = logging.getLogger(__name__)

class GeolocationUtils:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="solar_analysis_app")
        
    def reverse_geocode(self, latitude, longitude):
        """Convert coordinates to address"""
        try:
            location = self.geolocator.reverse(f"{latitude}, {longitude}")
            return location.address if location else "Address not found"
        except Exception as e:
            logger.error(f"Reverse geocoding error: {e}")
            return self.get_address_from_osm(latitude, longitude)
    
    def get_address_from_osm(self, lat, lon):
        """Fallback to OpenStreetMap API"""
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if 'address' in data:
                address = data.get('display_name', 'Address not found')
                return address
            return "Address not found"
        except:
            return f"Location: {lat:.4f}, {lon:.4f}"
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in km"""
        point1 = (lat1, lon1)
        point2 = (lat2, lon2)
        return geodesic(point1, point2).km
    
    def get_elevation(self, latitude, longitude):
        """Get elevation for coordinates"""
        try:
            url = f"https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}"
            response = requests.get(url, timeout=5)
            data = response.json()
            return data['results'][0]['elevation']
        except:
            return 0
    
    def validate_coordinates(self, latitude, longitude):
        """Validate if coordinates are within India"""
        # India bounds approximately
        india_bounds = {
            'min_lat': 6.0, 'max_lat': 38.0,
            'min_lon': 68.0, 'max_lon': 98.0
        }
        
        return (india_bounds['min_lat'] <= latitude <= india_bounds['max_lat'] and
                india_bounds['min_lon'] <= longitude <= india_bounds['max_lon'])