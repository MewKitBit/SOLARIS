import pandas as pd
import pvlib

# TODO: Expand to support static and tracking mounts, this is only static for now.
def fetch_solar_data(latitude: float, longitude: float, timezone: str, initial_date: pd.Timestamp,
                     final_date: pd.Timestamp, tilt: float, azimuth: float) -> pd.DataFrame:
    """
    Fetches historical in-plane irradiance and weather data via PVGIS for a given location,
    panel geometry, and timeframe.

    PVGIS returns irradiance already projected onto the panel surface (poa_direct,
    poa_sky_diffuse, poa_ground_diffuse), so no transposition step is needed downstream.

    :param latitude: Decimal latitude of the location.
    :param longitude: Decimal longitude of the location.
    :param timezone: IANA timezone string used to localise the returned DatetimeIndex.
    :param initial_date: Start of the requested period (inclusive).
    :param final_date: End of the requested period (inclusive).
    :param tilt: Panel tilt angle in degrees from horizontal.
    :param azimuth: Panel azimuth angle in degrees (180 = south-facing).
    :return: ``pd.DataFrame`` with irradiance and weather columns at hourly resolution,
             localised to ``timezone``. Returns an empty DataFrame if the PVGIS request fails.
    """

    weather = pd.DataFrame()

    try:
        weather, inputs = pvlib.iotools.get_pvgis_hourly(
            latitude=latitude,
            longitude=longitude,
            start=initial_date.year,
            end=final_date.year,
            components=True,
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            pvcalculation=False
        )
    except Exception as e:
        print(f"Error fetching data from PVGIS: {e}")
        return weather

    weather.index = weather.index.tz_convert(timezone)
    weather = weather.loc[initial_date:final_date]

    return weather
