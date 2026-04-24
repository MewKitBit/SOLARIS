import requests
import pandas as pd

openmeteo_url = "https://archive-api.open-meteo.com/v1/archive"

def _resample_volume(series: pd.Series, resolution: str, target: int, source: int) -> pd.Series:
    """
    Upsamples a precipitation series to a finer resolution by forward-filling and
    scaling values proportionally to the interval ratio.

    For example, hourly data resampled to 15-minute resolution will distribute each
    hour's total evenly across its four sub-intervals (each bucket gets 0.25× the
    original value).

    :param series: Hourly precipitation series (mm).
    :param resolution: Target pandas-style frequency string (e.g. ``'15min'``, ``'30min'``).
    :param target: Target interval size in nanoseconds.
    :param source: Source interval size in nanoseconds (the native hourly interval).
    :return: Resampled series scaled to the new interval width.
    """

    scale_factor = target / source
    return series.resample(resolution).ffill() * scale_factor

def fetch_rain_volume(latitude: float, longitude: float, start_date: str, end_date: str,
                      resolution: str = "1h") -> pd.Series:
    """
    Fetches hourly precipitation data (mm) from Open-Meteo for the inclusive range
    ``[start_date, end_date]`` and resamples to the requested resolution if needed.

    Open-Meteo natively provides hourly data. Coarser resolutions (e.g. ``'D'``, ``'ME'``)
    are aggregated with ``sum``; finer resolutions (e.g. ``'15min'``) are produced by
    forward-filling and proportional scaling via :func:`_resample_volume`.

    :param latitude: Decimal latitude of the location.
    :param longitude: Decimal longitude of the location.
    :param start_date: Start of the requested period (inclusive), as ``YYYY-MM-DD`` string.
    :param end_date: End of the requested period (inclusive), as ``YYYY-MM-DD`` string.
    :param resolution: Pandas-style frequency string for the output interval (default ``'1h'``).
    :return: ``pd.Series`` named ``precipitation_mm`` with a UTC ``DatetimeIndex``.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation",
        "timezone": "UTC",
    }

    response = requests.get(openmeteo_url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    times = pd.to_datetime(payload["hourly"]["time"], utc=True)
    values = payload["hourly"]["precipitation"]

    series = pd.Series(values, index=times, dtype=float, name="precipitation_mm")
    series = series.loc[start_date:end_date]

    target_nanos = pd.tseries.frequencies.to_offset(resolution).nanos
    hourly_nanos = pd.tseries.frequencies.to_offset("1h").nanos

    if target_nanos > hourly_nanos:
        series = series.resample(resolution).sum()
    elif target_nanos < hourly_nanos:
        series = _resample_volume(series, resolution, target_nanos, hourly_nanos)

    return series