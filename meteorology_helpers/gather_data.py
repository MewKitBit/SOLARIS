import argparse
import pandas as pd

# Workaround to tomllib not being included in Python versions under 3.11
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from pathlib import Path
from precipitation import fetch_rain_volume
from solar_position import fetch_solar_data

_CONFIG_PATH = Path(__file__).parent/"config.toml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)

def _gather_precipitation(cfg: dict, output_dir: Path) -> None:
    loc = cfg["location"]
    period = cfg["period"]

    series = fetch_rain_volume(
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        start_date=period["start_date"],
        end_date=period["end_date"],
    )

    out = output_dir/"precipitation.csv"
    series.to_csv(out, header=True)
    print(f"Precipitation saved -> {out}")

def _gather_solar(cfg: dict, output_dir: Path) -> None:
    loc = cfg["location"]
    period = cfg["period"]
    solar = cfg["solar"]
    timezone = loc["timezone"]

    df = fetch_solar_data(
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        timezone=timezone,
        initial_date=pd.Timestamp(period["start_date"], tz=timezone),
        final_date=pd.Timestamp(period["end_date"], tz=timezone),
        tilt=solar["tilt"],
        azimuth=solar["azimuth"],
    )

    if df.empty:
        print("Solar data fetch failed, nothing saved.")
        return

    out = output_dir/"solar_position.csv"
    df.to_csv(out)
    print(f"Solar data saved -> {out}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch meteorological data from remote sources.")
    parser.add_argument("--prec", action="store_true", help="Fetch precipitation data from Open-Meteo.")
    parser.add_argument("--solpos", action="store_true", help="Fetch solar irradiance data from PVGIS.")
    parser.add_argument("--output_dir", type=Path, default=Path("."), metavar="DIR",
                        help="Directory where output CSV files are written (default: current directory).")
    args = parser.parse_args()

    if not args.prec and not args.solpos:
        parser.error("At least one of --prec or --solpos must be specified.")

    cfg = _load_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.prec:
        _gather_precipitation(cfg, args.output_dir)
    if args.solpos:
        _gather_solar(cfg, args.output_dir)

if __name__ == "__main__":
    main()
