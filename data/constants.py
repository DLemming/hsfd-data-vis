# ---------------------------------
# ----------- Sidebar -------------
# ---------------------------------

MIN_SAMPLES = 1_000
MAX_SAMPLES = 9_900
DEFAULT_SAMPLES = 1_000
HEALTHY_ONLY = True
USE_SUBSET = True


# ---------------------------------
# --------- Coordinates -----------
# ---------------------------------

# NYC Bounds
LATITUDE_MIN = 40.5
LATITUDE_MAX = 41.0
LONGITUDE_MIN = -74.3
LONGITUDE_MAX = -73.6

# Typical NYC locations for taxi pickup/dropoff
NYC_LOCATIONS = {
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport": (40.7769, -73.8740),
    "Times Square": (40.7580, -73.9855),
    "Central Park": (40.7851, -73.9683),
    "Empire State Building": (40.7484, -73.9857),
    "Wall Street": (40.7066, -74.0090),
    "Brooklyn Bridge": (40.7061, -73.9969),
    "Grand Central Terminal": (40.7527, -73.9772),
    "Yankee Stadium": (40.8296, -73.9262),
    "Coney Island": (40.5749, -73.9850),
}


# ---------------------------------
# -------- Health Checks ----------
# ---------------------------------

ALLOWED_SELECTION_HEALTH = [
    "VendorID",
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude","dropoff_latitude",
    "payment_type","fare_amount","tip_amount","tolls_amount","improvement_surcharge","total_amount"
]

CUSTOM_CHECKS = {
    "passenger_count": lambda s: s.notnull() & (s > 0),
    "trip_distance": lambda s: s.notnull() & (s > 0),
    "pickup_longitude": lambda s: s.notnull() & (s > LONGITUDE_MIN) & (s < LONGITUDE_MAX),
    "pickup_latitude": lambda s: s.notnull() & (s > LATITUDE_MIN) & (s < LATITUDE_MAX),
    "dropoff_longitude": lambda s: s.notnull() & (s > LONGITUDE_MIN) & (s < LONGITUDE_MAX),
    "dropoff_latitude": lambda s: s.notnull() & (s > LATITUDE_MIN) & (s < LATITUDE_MAX),
    "fare_amount": lambda s: s.notnull() & (s > 0),
    "mta_tax": lambda s: s.notnull() & (s >= 0),
    "tip_amount": lambda s: s.notnull() & (s >= 0),
    "tolls_amount": lambda s: s.notnull() & (s >= 0),
    "improvement_surcharge": lambda s: s.notnull() & (s >= 0),
    "total_amount": lambda s: s.notnull() & (s > 0),
}

UNIQUE_THRESH = 20