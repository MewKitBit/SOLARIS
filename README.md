# SOLARIS
**Photovoltaic module simulator with fault injection and composition**

SOLARIS simulates fleets of PV modules over multi-year time series and enables injection of individual faults and degradation modes, each with their own onset time and progression models. 

Every run produces two datasets: 
1. The I-V characteristic points of every panel at every time step
2. A ground-truth record of which effect hit which panel and when 

That pairing is what fault-detection and state-of-health algorithms need for training and benchmarking, and what is hard to get from real installations.

Fault detection in PV is an active research field, but public annotated data is scarce:

- **Experimental datasets** (e.g. GPVS-Faults, PVEL-AD) are expensive to produce and tied to one installation, one climate and a closed set of faults.
- **Established simulators** (PVsyst, SAM, pvlib) model healthy behaviour with high fidelity but treat degradation as an aggregate percentage loss. They cannot inject a specific failure mode with a realistic temporal progression, and they do not produce labels.

SOLARIS sits in that gap, bridging physically grounded simulation on top of [pvlib](https://github.com/pvlib/pvlib-python) with a modular effect layer built specifically to produce labelled, reproducible, extensible synthetic data.
## Features
- **Physics-based core**: Single-diode model (De Soto / CEC parametrization) solved through [pvlib](https://github.com/pvlib/pvlib-python), with module parameters from the CEC database.
- **Institutional weather data**: Irradiance and solar position from [PVGIS](https://pvgis.com/en) (EC Joint Research Centre), auxiliary meteorological variables from [Open-Meteo](https://open-meteo.com) (ERA5 reanalysis).
- **Pluggable effects**: Every fault or degradation mode implements one small abstract interface and declares which physical parameters it perturbs. No changes to the core are needed to add one.
- **Effect composition**:Any number of effects (including several instances of the same one) can act on the same panel simultaneously, and a composition engine combines them per parameter.
- **Ground-truth attribution**: Onset timestamp of every effect on every panel, written alongside the observables.
- **Reproducible**: Each `(panel, effect)` pair gets an independent, deterministically derived RNG stream from a single master seed. Using the same seed produces the same output, regardless of how the fleet is chunked.
- **Parallel and memory-bounded**: `multiprocessing.Pool` execution with chunk sizes derived from a user-set RAM cap.
- **Single declarative config**: One TOML file drives both data acquisition and simulation. Config is validated up front and all errors are reported together, before any computation starts.
- **Columnar output**: Partitioned `Parquet` datasets readable with a single `pd.read_parquet(dir)`.

---
## Installation
Requires Python 3.11+ (SOLARIS reads its config with `tomllib`; on older versions install `tomli` as a fallback).

```bash
git clone https://github.com/MewKitBit/SOLARIS.git
cd SOLARIS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `pvlib`, `numpy`, `pandas`, `pyarrow`, `requests`, `tqdm`.

## Quick start
SOLARIS runs in two decoupled phases, both driven by the same config file.

**1. Acquire input data** (once per location/period; the result can be reused across simulations):

```bash
python meteorology_helpers/gather_data.py -c --cfg solaris.toml --output-dir ./inputs/
```

**2. Run the simulation:**

```bash
python main.py --cfg solaris.toml
```

**3. Read the results:**

```python
import pandas as pd

obs  = pd.read_parquet("output/observables")   # I-V points per (panel_id, timestamp)
attr = pd.read_parquet("output/attribution")   # effect onset times per panel_id
```

---
## Effects, faults and degradation
SOLARIS uses _effect_ as the umbrella term:

- An **effect** is any deviation of a panel's response from its nominal behaviour.
- A **fault** is a conditional effect whose onset is governed by a statistical distribution and the environment, and may never occur.
- A **degradation** is gradual and always present from the first timestep.

Effects never touch the I-V curve directly. They declare which inputs they perturb and return a per-timestep modifier for each. The valid axes and their composition rules are:

| Axis   | Quantity                 | Composition    | Value                      |
| ------ | ------------------------ | -------------- | -------------------------- |
| `G`    | Effective irradiance     | Multiplicative | fractional loss r ∈ [0, 1] |
| `I_L`  | Photocurrent             | Multiplicative | fractional loss r ∈ [0, 1] |
| `R_sh` | Shunt resistance         | Multiplicative | fractional loss r ∈ [0, 1] |
| `T`    | Cell temperature         | Additive       | δ in °C                    |
| `R_s`  | Series resistance        | Additive       | δ in Ω                     |
| `I_0`  | Diode saturation current | Additive       | δ in A                     |

When several effects act on the same axis, multiplicative losses combine as `baseline · Π(1 − rᵢ)` and additive deltas as `baseline + Σδᵢ`. 

Modifiers are applied in two stages, so that environmental effects propagate correctly through the parameter calculation:
1. `G` and `T` modifiers are applied to the baseline effective irradiance and cell temperature.
2. The five SDM parameters are computed from the modified conditions via `calcparams_cec`.
3. `I_L`, `I_0`, `R_s` and `R_sh` modifiers are applied to those parameters.
4. The single-diode equation is solved for the I-V characteristic points.

### Optimisations
- **Per-chunk baseline**: Effective irradiance (including the incidence-angle modifier) and cell temperature are computed once per chunk.
- **Dark-row masking**: With zero irradiance every I-V point is zero regardless of effects, and since irradiance modifiers are multiplicative, a dark row stays dark for every panel. These rows (≈50 % of a year) are masked out before solving and refilled with zeros afterwards.
- **Preallocated outputs**: Each panel writes into its own contiguous block of rows, avoiding a final concatenation that would double peak memory.
---
## Configuration
See example TOML file.

| Section              | Keys                                                                   | Notes                                                                                                                            |
| -------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `[location]`         | `latitude`, `longitude`, `timezone`                                    | Decimal degrees, positive = north/east. IANA timezone.                                                                           |
| `[period]`           | `start_date`, `end_date`                                               | `YYYY-MM-DD`, both inclusive.                                                                                                    |
| `[runtime]`          | `master_seed`, `num_panels`, `memory_cap_gb`, `num_workers`            | `num_workers = 1` runs single-process; `> 1` uses a process pool. The memory cap is approximate and split evenly across workers. |
| `[paths]`            | `intake_file`, `output_dir`                                            | Input CSV and output root.                                                                                                       |
| `[singlediode]`      | `method`                                                               | `NEWTON`, `LAMBERTW`, `BRENTQ` or `CHANDRUPATLA`. **`NEWTON` is strongly recommended** for large runs.                           |
| `[temperature]`      | `model`                                                                | SAPM or PVsyst variants, e.g. `SAPM_OPEN_RACK_GLASS`.                                                                            |
| `[incident_angle]`   | `model` + per-model glass params                                       | `PHYSICAL` (`n`, `K`, `L`), `ASHRAE` (`b`) or `MARTIN_RUIZ` (`a_r`). Only the set matching `model` is used.                      |
| `[module]`           | `alpha_sc`, `a_ref`, `I_L_ref`, `I_o_ref`, `R_sh_ref`, `R_s`, `Adjust` | The seven CEC reference parameters, verbatim from the CEC database entry.                                                        |
| `[mounting]`         | `surface_tilt`, `surface_azimuth`                                      | Fixed mount only. Azimuth in degrees clockwise from north.                                                                       |
| `[[effects.<type>]]` | `name` + model-specific keys                                           | See below.                                                                                                                       |
### Declaring effects
Effects are nested arrays of tables. The outer key must match the effect class's `type` attribute; each inner entry is one instance, and its keys are passed straight to the class constructor. The `name` key must be unique across the whole file and becomes the attribution column prefix.

```toml
[[effects.soiling_variant]]
name                     = "soiling"
max_soiling_effect       = 0.3      # asymptotic cap on irradiance loss (0.3 = 30 %)
avg_hourly_progression   = 1.0e-5   # positive drift of the random walk
std_deviation            = 3.0e-5   # per-step noise; negative draws model cleaning
starting_median_fraction = 0.25     # optional; median initial level as fraction of the cap
starting_shape_sigma     = 0.5      # optional; lognormal spread of initial levels
```

Removing an entry disables that effect, whereas duplicating it with a different `name` and parameters runs both variants composed.

### Validation
Before anything runs, `validate_config_file` checks required sections and keys, `[runtime]` coherence, model identifiers against the code enums, the glass parameters required by the chosen IAM model, the CEC parameters, and the effect declarations (registered type, array-of-tables form, unique names). Every violation is collected and reported in a single exception. Physical plausibility of the values is **not** checked; that stays the user's responsibility.

## Input data format
`intake_file` is a CSV whose first column, `time`, is a timezone-aware timestamp (e.g. `2005-01-01 00:00:00+00:00`) used as the index. Required columns:

|Column|Description|Unit|
|---|---|---|
|`poa_direct`|Direct plane-of-array irradiance|W/m²|
|`poa_sky_diffuse`|Sky-diffuse POA irradiance|W/m²|
|`poa_ground_diffuse`|Ground-reflected POA irradiance|W/m²|
|`poa_global`|Total POA irradiance|W/m²|
|`temp_air`|Ambient air temperature|°C|
|`wind_speed`|Wind speed|m/s|
|`solar_zenith`|Apparent solar zenith angle|°|
|`solar_azimuth`|Solar azimuth angle|°|

Any additional column (such as relative humidity) is passed through to effect models that need it. Helpers can be entirely skipped and data can be supplied from another source, as long as it matches this format.
### Data acquisition helper
`meteorology_helpers/gather_data.py` takes `--cfg` (config path) and `--output-dir` (created if missing where same-named files are overwritten), plus one mode flag:

| Flag       | Fetches                                                    | Writes                               |
| ---------- | ---------------------------------------------------------- | ------------------------------------ |
| `--solpos` | PVGIS irradiance and solar position                        | `solar_position.csv`                 |
| `--omet`   | Open-Meteo fields listed in `meteorology_helpers/meteo.py` | `meteorology.csv`                    |
| `-c`       | Both, then merges them                                     | both of the above + `input_data.csv` |

To request extra atmospheric variables for a new effect, add the fields to the Open-Meteo call in `meteo.py`. Note that the free [Open-Meteo](https://open-meteo.com) API is licensed for non-commercial use.
## Output format

```
output_dir/
├── observables/
│   ├── observables_00000000_00000417.parquet
│   ├── observables_00000417_00000834.parquet
│   └── ...
└── attribution/
    ├── attribution_00000000_00000417.parquet
    └── ...
```

Each chunk writes one part-file per dataset, named after the global panel range it contains (`<start>_<end>`, zero-padded to eight digits, end exclusive).

**`observables`** is indexed by `(panel_id, timestamp)`, `num_panels × num_timesteps` rows, all `float64`:

|Column|Quantity|Unit|
|---|---|---|
|`i_sc`|Short-circuit current|A|
|`v_oc`|Open-circuit voltage|V|
|`i_mp`|Current at maximum power point|A|
|`v_mp`|Voltage at maximum power point|V|
|`p_mp`|Maximum power|W|
|`i_x`|Current at V = V_oc / 2|A|
|`i_xx`|Current at V = (V_oc + V_mp) / 2|A|

**`attribution`** is one row per `panel_id`, one `<name>_onset` column per registered effect, holding the timestamp at which that effect activated on that panel, or `NaT` if it never did within the simulated period.

---
## Adding an effect

1. **Subclass** `AbstractBaselineFailure` in `module_pipeline/failure_types/`:

```python
from numpy.random import default_rng
from pandas import DataFrame

from .abstract_baseline_failure import AbstractBaselineFailure


class LinearSeriesResistanceDrift(AbstractBaselineFailure):
    type = "rs_drift"
    affected_columns = ("R_s",)      # subset of G, I_L, R_sh, T, R_s, I_0

    def __init__(self, env_params: DataFrame, ohms_per_step: float, name: str):
        super().__init__(name=name, env_params=env_params)
        self.ohms_per_step = ohms_per_step

    def compute_time_to_onset(self, seed: int) -> int:
        # First active step for this panel. If over len(series) it means "never occurs".
        rng = default_rng(seed)
        return int(rng.integers(0, len(self.env_params.index)))

    def compute_progression(self, seed: int, start_step: int, end_step: int) -> DataFrame:
        index = self.env_params.index[start_step:end_step]
        steps = range(1, len(index) + 1)
        return DataFrame({"R_s": [s * self.ohms_per_step for s in steps]}, index=index)
```

2. **Register** the class by adding it to `FAILURE_CLASSES` in `module_pipeline/failure_types/__init__.py`.

3. **Declare** one or more instances in the config:

```toml
[[effects.rs_drift]]
name          = "contact_corrosion"
ohms_per_step = 1.0e-7
```

Always seed randomness from the `seed` argument (never a global RNG); this is what guarantees reproducibility and per-panel independence. `simple_soiling.py` is the reference implementation: lognormal initial soiling, then a bounded random walk with positive drift acting on `G`.

---
## Limitations
- One module type and one fixed orientation per run; no single-axis trackers, no string-level aggregation.
- Effects act only through the six SDM-input axes. Failures that affect power without altering those inputs (such as MPPT faults) are not yet expressible.
- Composition rules (product of multiplicative losses, sum of additive deltas) are a first-order model and have not been validated against field data.

## Future improvements
- Additional mount types and string-level aggregation.
- Effects outside the SDM parameters (MPPT and inverter-side faults).
- Refine composition rules with domain experts.
- Distributed execution across multiple machines.

Contributions of fault and degradation models from PV reliability experts are especially welcome: SOLARIS deliberately leaves model physics to the user so that the core can serve as a common harness.

---
## Citation

If you use SOLARIS in your work, please cite the thesis:

```bibtex
@thesis{onamarisescu2026solaris,
  author      = {Ona Marisescu, Ionut Alexandru},
  title       = {{SOLARIS}: Simulador de m{\'o}dulos fotovoltaicos con inyecci{\'o}n y composici{\'o}n de fallos},
  type        = {Bachelor's thesis (Proyecto Fin de Grado)},
  institution = {E.T.S. de Ingenier{\'i}a de Sistemas Inform{\'a}ticos, Universidad Polit{\'e}cnica de Madrid},
  year        = {2026},
  month       = jul,
  url         = {https://github.com/MewKitBit/SOLARIS}
}
```

The full design rationale, state of the art and evaluation are in the thesis report (Spanish) available in this repository.
## License
Code released under the [MIT License](LICENSE). The thesis report is licensed separately under CC BY-NC-SA 4.0.
