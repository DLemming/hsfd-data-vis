# ---------------------------------
# ----------- Sidebar -------------
# ---------------------------------

SAMPLE_SIZE = 1_000
HEALTHY_ONLY = True
FULL_DATASET = False


# ---------------------------------
# --------- Coordinates -----------
# ---------------------------------

# NYC Bounds
LATITUDE_MIN = 40.5
LATITUDE_MAX = 41.0
LONGITUDE_MIN = -74.3
LONGITUDE_MAX = -73.6


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