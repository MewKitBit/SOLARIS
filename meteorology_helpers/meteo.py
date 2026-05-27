import requests
import pandas as pd

openmeteo_url = "https://archive-api.open-meteo.com/v1/archive"

def fetch_omet(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches hourly precipitation (mm), surface pressure (hPa), and relative humidity (%)
    from Open-Meteo for the inclusive range ``[start_date, end_date]``.

    :param latitude: Decimal latitude of the location.
    :param longitude: Decimal longitude of the location.
    :param start_date: Start of the requested period (inclusive), as ``YYYY-MM-DD`` string.
    :param end_date: End of the requested period (inclusive), as ``YYYY-MM-DD`` string.
    :return: ``pd.DataFrame`` with columns ``precipitation_mm``, ``pressure``, and
             ``relative_humidity_pct``, indexed by a UTC ``DatetimeIndex``.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation,surface_pressure,relative_humidity_2m",
        "timezone": "UTC",
    }

    response = requests.get(openmeteo_url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    times = pd.to_datetime(payload["hourly"]["time"], utc=True)

    df = pd.DataFrame({
        "precipitation_mm": payload["hourly"]["precipitation"],
        "pressure": payload["hourly"]["surface_pressure"],
        "relative_humidity_pct": payload["hourly"]["relative_humidity_2m"],
    }, index=times, dtype=float)

    return df.loc[start_date:end_date]
