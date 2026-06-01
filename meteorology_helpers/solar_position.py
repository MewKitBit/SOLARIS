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
    Solar position (``solar_zenith``, ``solar_azimuth``) is derived here via
    ``pvlib.solarposition`` and ``poa_global`` is summed from the three POA components, so
    the returned frame is a complete, self-describing input for the simulation step.

    :param latitude: Decimal latitude of the location.
    :param longitude: Decimal longitude of the location.
    :param timezone: IANA timezone string used to localise the returned DatetimeIndex.
    :param initial_date: Start of the requested period (inclusive).
    :param final_date: End of the requested period (inclusive).
    :param tilt: Panel tilt angle in degrees from horizontal.
    :param azimuth: Panel azimuth angle in degrees (180 = south-facing).
    :return: ``pd.DataFrame`` with irradiance and weather columns at hourly resolution,
             localised to ``timezone``, plus the derived ``solar_zenith``, ``solar_azimuth``,
             and ``poa_global`` columns. Returns an empty DataFrame if the PVGIS request fails.
    """

    try:
        data, metadata = pvlib.iotools.get_pvgis_hourly(
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
        return pd.DataFrame()

    data.index = data.index.tz_convert(timezone)

    # PVGIS reports solar_elevation but not azimuth, and AOI needs both. Solar position is a
    # property of this location and these timestamps, so deriving it here (rather than in the
    # simulation) keeps it pinned to the same provenance as the irradiance and removes any
    # chance of lat/lon drift between the gather and simulation configs.
    solar_position = pvlib.solarposition.get_solarposition(data.index, latitude, longitude)
    data['solar_zenith'] = solar_position['apparent_zenith']
    data['solar_azimuth'] = solar_position['azimuth']
    data['poa_global'] = data['poa_direct'] + data['poa_sky_diffuse'] + data['poa_ground_diffuse']

    # final_date is midnight at the start of the last requested day; extend the upper bound to
    # the end of that day so the inclusive slice keeps it whole, matching fetch_omet's
    # whole-day string slice. Without this, the last day's solar columns come back empty.
    inclusive_end = final_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    data = data.loc[initial_date:inclusive_end]

    return data
