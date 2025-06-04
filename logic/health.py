import pandas as pd

CUSTOM_CHECKS = {
    "passenger_count": lambda s: s.notnull() & (s != 0),
    "trip_distance": lambda s: s.notnull() & (s >= 0),
    "pickup_longitude": lambda s: s.notnull() & (s > -79) & (s < -71),
    "pickup_latitude": lambda s: s.notnull() & (s > 40) & (s < 46),
    "dropoff_longitude": lambda s: s.notnull() & (s > -79) & (s < -71),
    "dropoff_latitude": lambda s: s.notnull() & (s > 40) & (s < 46),
    "fare_amount": lambda s: s.notnull() & (s >= 0),
    "mta_tax": lambda s: s.notnull() & (s >= 0),
    "tip_amount": lambda s: s.notnull() & (s >= 0),
    "tolls_amount": lambda s: s.notnull() & (s >= 0),
    "improvement_surcharge": lambda s: s.notnull() & (s >= 0),
    "total_amount": lambda s: s.notnull() & (s >= 0),
}

def compute_health(
    df: pd.DataFrame,
    columns: list[str],
) -> dict:
    total = len(df)
    checks = []

    for col in columns:
        if CUSTOM_CHECKS and col in CUSTOM_CHECKS:
            checks.append(CUSTOM_CHECKS[col](df[col]))
        else:
            checks.append(df[col].notnull())

    if not checks:
        healthy_mask = pd.Series([False] * total, index=df.index)
    else:
        healthy_mask = checks[0]
        for mask in checks[1:]:
            healthy_mask &= mask

    healthy_count = healthy_mask.sum()
    unhealthy_count = total - healthy_count
    healthy_percent = round(100 * healthy_count / total, 2) if total else 0

    return {
        "healthy_mask": healthy_mask,
        "healthy_count": healthy_count,
        "unhealthy_count": unhealthy_count,
        "healthy_percent": healthy_percent,
        "total": total
    }