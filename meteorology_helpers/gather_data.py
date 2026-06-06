import argparse
import pandas as pd

# Workaround to tomllib not being included in Python versions under 3.11
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from pathlib import Path
from meteo import fetch_omet
from solar_position import fetch_solar_data

def _load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _gather_omet(cfg: dict, output_dir: Path) -> pd.DataFrame:
    loc = cfg["location"]
    period = cfg["period"]

    df = fetch_omet(
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        start_date=period["start_date"],
        end_date=period["end_date"],
    )

    df.index += pd.Timedelta(minutes=10)
    out = output_dir/"meteorology.csv"
    df.to_csv(out)
    print(f"Precipitation, pressure, & humidity saved -> {out}")
    return df


def _gather_solar(cfg: dict, output_dir: Path) -> pd.DataFrame:
    loc = cfg["location"]
    period = cfg["period"]
    mounting = cfg["mounting"]
    timezone = loc["timezone"]

    df = fetch_solar_data(
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        timezone=timezone,
        initial_date=pd.Timestamp(period["start_date"], tz=timezone),
        final_date=pd.Timestamp(period["end_date"], tz=timezone),
        tilt=mounting["surface_tilt"],
        azimuth=mounting["surface_azimuth"],
        )

    if not df.empty:
        out = output_dir/"solar_position.csv"
        df.to_csv(out)
        print(f"Solar data saved -> {out}")

    return df


def _gather_combined(cfg: dict, output_dir: Path) -> None:
    solar_df = _gather_solar(cfg, output_dir)
    if solar_df.empty:
        return

    omet_df = _gather_omet(cfg, output_dir)

    pd.concat([omet_df, solar_df], axis=1).to_csv(output_dir/"input_data.csv")
    print(f"Combined input data saved -> {output_dir/'input_data.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch meteorological data from remote sources.")
    parser.add_argument("--omet", action="store_true",
                        help="Fetch precipitation and pressure data from Open-Meteo.")
    parser.add_argument("--solpos", action="store_true",
                        help="Fetch solar irradiance data from PVGIS.")
    parser.add_argument("-c", action="store_true",
                        help="Fetch both sources and write meteorology.csv, solar_position.csv, and input_data.csv.")
    parser.add_argument("--cfg", type=Path, required=True, metavar="FILE",
                        help="Path to the unified SOLARIS config TOML (shared with the simulation).")
    parser.add_argument("--output-dir", type=Path, default=Path("."), metavar="DIR",
                        help="Directory where output CSV files are written (default: current directory).")
    args = parser.parse_args()

    if not args.omet and not args.solpos and not args.c:
        parser.error("At least one of --omet, --solpos, or -c must be specified.")
    if not args.cfg.is_file():
        parser.error(f"Config file not found: {args.cfg}")

    cfg = _load_config(args.cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.c:
        _gather_combined(cfg, args.output_dir)
    else:
        if args.omet:
            _gather_omet(cfg, args.output_dir)
        if args.solpos:
            _gather_solar(cfg, args.output_dir)


if __name__ == "__main__":
    main()
