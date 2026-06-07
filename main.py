import argparse

from pandas import DataFrame, read_csv
from pathlib import Path

from module_pipeline import effect_generator
from module_pipeline.data_generator import generate_data
from module_pipeline.enums import IncidentAngleModel, SingleDiodeMethod, TemperatureModel
from module_pipeline.failure_types import FAILURE_CLASSES

# Workaround to tomllib not being included in Python versions under 3.11
try:
    import tomllib
except ImportError:
    import tomli as tomllib

# IAM glass parameters consumed per incident-angle model; the chosen model's set must be present.
_IAM_GLASS_PARAMS = {'PHYSICAL': ('n', 'K', 'L'), 'ASHRAE': ('b',), 'MARTIN_RUIZ': ('a_r',)}

# TODO: Add logging and console outputs to every step, otherwise the process is a black box until it finishes or fails
# TODO: Maybe even add a progress bar for the single_diode chunk result writing with sub-bars for each worker
def orchestrator(config: dict) -> None:
    """
    Runs the full simulation from a validated config: loads the inputs, builds and configures the
    effects, then drives ``generate_data`` over memory-capped chunks, writing each chunk's
    observables and attribution frames to parquet under ``[paths] output_dir``.

    Assumes ``config`` has already passed ``validate_config_file``. Chunks are independent units of
    work that each worker processes one at a time.

    :param config: parsed and validated TOML config.
    """

    runtime = config['runtime']

    env_params, solar_positions = _load_inputs(config['paths']['intake_file'])
    num_timesteps = len(env_params)

    module_params = _build_module_params(config)
    temp_model = TemperatureModel[config['temperature']['model']]
    iam_model = IncidentAngleModel[config['incident_angle']['model']]
    method = SingleDiodeMethod[config['singlediode']['method'].upper()]

    effects = _build_effects(config, env_params)
    effect_generator.configure(effects, runtime['master_seed'], num_timesteps)
    effect_generator.validate_effects()

    output_dir = Path(config['paths']['output_dir'])
    # Observables and attribution each get their own subdirectory so each is a single-schema
    # parquet dataset a consumer can read whole with read_parquet(dir)
    observables_dir = output_dir / 'observables'
    attribution_dir = output_dir / 'attribution'
    observables_dir.mkdir(parents=True, exist_ok=True)
    attribution_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = _compute_chunk_size(runtime['memory_cap_gb'], num_timesteps)
    chunks = _chunk_ranges(runtime['num_panels'], chunk_size)

    # TODO: parallelize chunks across runtime['num_workers'] cores; single-core for now.
    for panel_ids in chunks:
        observables, attribution = generate_data(
            module_params, env_params, solar_positions, temp_model, iam_model, panel_ids, method,
        )
        _write_chunk(observables_dir, attribution_dir, panel_ids, observables, attribution)


def _load_inputs(intake_file: str) -> tuple[DataFrame, DataFrame]:
    """
    Reads the intake CSV and splits it into the two frames ``generate_data`` consumes.

    The intake file is produced by ``meteorology_helpers/gather_data.py -c`` and is indexed by a
    tz-aware timestamp. ``solar_positions`` is the ``solar_azimuth``/``solar_zenith`` pair used for
    the angle-of-incidence calc; ``env_params`` is the full frame (irradiance, weather, and humidity
    columns the temperature model and effects read).

    :param intake_file: path to the combined intake CSV.
    :return: tuple ``(env_params, solar_positions)``.
    """

    data = read_csv(intake_file, index_col=0, parse_dates=True)
    solar_positions = data[['solar_azimuth', 'solar_zenith']]
    return data, solar_positions


def _build_module_params(config: dict) -> dict:
    """
    Flattens the module-related TOML sections into the single dict ``generate_data`` expects.

    Merges the ``[module]`` CEC reference parameters, the ``[mounting]`` array geometry, and the
    ``[incident_angle]`` glass parameters that match the chosen IAM model.

    :param config: parsed TOML config.
    :return: flat dict of CEC params, mounting geometry, and the active IAM glass params.
    """

    module_params = {**config['module'], **config['mounting']}

    incident_angle = config['incident_angle']
    for key in _IAM_GLASS_PARAMS[incident_angle['model']]:
        module_params[key] = incident_angle[key]

    return module_params


def _build_effects(config: dict, env_params: DataFrame) -> list:
    """
    Instantiates the configured failure objects by walking the class registry.

    For each class in ``FAILURE_CLASSES``, looks up its ``[[effects.<type>]]`` entries and builds
    one instance per entry, passing the entry's keys as constructor kwargs. Classes with no matching
    TOML section contribute nothing.

    :param config: parsed TOML config.
    :param env_params: environmental frame each effect holds for its acceleration math.
    :return: list of instantiated failure objects, in registry-then-config order.
    """

    effects_config = config.get('effects', {})
    effects = []
    for cls in FAILURE_CLASSES:
        for entry in effects_config.get(cls.type, []):
            effects.append(cls(env_params, **entry))

    return effects


# Observables dominate resident memory: seven float64 I-V points per (panel, timestep).
_OBSERVABLE_BYTES = 8 * 7
# Headroom reserved over the observables frame for the MultiIndex, singlediode scratch, and the OS.
_CHUNK_SAFETY_FACTOR = 0.5


def _compute_chunk_size(memory_cap_gb: float, num_timesteps: int) -> int:
    """
    Derives how many panels fit in one chunk under the configured memory cap.

    Sizes against the observables frame (``num_timesteps`` rows times seven float64 columns per
    panel), scaled by a safety factor reserving room for the index, the single-diode solver's
    scratch space, and other live objects. Returns at least one panel so a tight cap still makes
    progress.

    :param memory_cap_gb: per-chunk memory budget in gigabytes, from ``[runtime]``.
    :param num_timesteps: length of the simulation time axis.
    :return: chunk size in panels (>= 1).
    """

    bytes_per_panel = num_timesteps * _OBSERVABLE_BYTES
    budget_bytes = memory_cap_gb * 1e9 * _CHUNK_SAFETY_FACTOR
    return max(1, int(budget_bytes // bytes_per_panel))


def _chunk_ranges(num_panels: int, chunk_size: int) -> list[range]:
    """
    Partitions ``range(num_panels)`` into contiguous global-id chunks of at most ``chunk_size``.

    :param num_panels: total fleet size.
    :param chunk_size: maximum panels per chunk.
    :return: list of ``range`` objects covering ``[0, num_panels)`` without overlap.
    """

    return [range(start, min(start + chunk_size, num_panels))
            for start in range(0, num_panels, chunk_size)]


def _write_chunk(observables_dir: Path, attribution_dir: Path, panel_ids: range,
                 observables: DataFrame, attribution: DataFrame) -> None:
    """
    Writes one chunk's observables and attribution frames as parquet part-files in their datasets.

    File names embed the chunk's global panel-id span so the part-files are ordered and
    non-colliding within each dataset directory.

    :param observables_dir: dataset directory the observables part-file is written to.
    :param attribution_dir: dataset directory the attribution part-file is written to.
    :param panel_ids: the chunk's global panel-id range.
    :param observables: the chunk's I-V observables frame.
    :param attribution: the chunk's per-panel onset frame.
    """

    span = f"{panel_ids.start:08d}_{panel_ids.stop:08d}"
    observables.to_parquet(observables_dir / f"observables_{span}.parquet")
    attribution.to_parquet(attribution_dir / f"attribution_{span}.parquet")

def validate_config_file(config: dict) -> None:
    """
    Quick pass over TOML config file to check its sanity. If misconfigured, raise an
    exception at the end indicating every issue found.

    Catches structural and obvious-value violations: missing sections, missing required
    keys, out-of-range runtime values, unknown enum values for ``[singlediode]``,
    ``[temperature]``, ``[incident_angle]``, IAM glass parameters that don't match the
    chosen IAM model, ``[[effects.<type>]]`` blocks whose ``<type>`` doesn't match a
    class in ``failure_types.FAILURE_CLASSES``, effect entries missing the required
    ``name`` key, and duplicate effect names. Does not validate physical sanity of CEC
    or effect parameters, nor per-effect constructor kwargs (Python raises ``TypeError``
    at instantiation if any are missing).

    :param config: TOML config dict result from read.
    :raises ValueError: when one or more violations are found, listing them together.
    """

    errors: list[str] = []

    required_sections = ('runtime', 'paths', 'singlediode', 'temperature',
                         'incident_angle', 'module', 'mounting')
    for section in required_sections:
        if section not in config:
            errors.append(f"missing required section [{section}]")
    if errors:
        # Without basic structure, deeper checks would all key-error.
        raise ValueError(_format_errors(errors))

    runtime = config['runtime']
    for key in ('master_seed', 'num_panels', 'memory_cap_gb', 'num_workers'):
        if key not in runtime:
            errors.append(f"[runtime] missing required key '{key}'")
    if runtime.get('num_panels', 1) < 1:
        errors.append(f"[runtime] num_panels must be >= 1 (got {runtime['num_panels']})")
    if runtime.get('memory_cap_gb', 1.0) <= 0:
        errors.append(f"[runtime] memory_cap_gb must be > 0 (got {runtime['memory_cap_gb']})")
    if runtime.get('num_workers', 1) < 1:
        errors.append(f"[runtime] num_workers must be >= 1 (got {runtime['num_workers']})")

    for key in ('intake_file', 'output_dir'):
        if key not in config['paths']:
            errors.append(f"[paths] missing required key '{key}'")

    if 'method' not in config['singlediode']:
        errors.append("[singlediode] missing required key 'method'")
    elif config['singlediode']['method'].upper() not in SingleDiodeMethod.__members__:
        valid = list(SingleDiodeMethod.__members__)
        errors.append(f"[singlediode] method='{config['singlediode']['method']}' not in {valid}")

    if 'model' not in config['temperature']:
        errors.append("[temperature] missing required key 'model'")
    elif config['temperature']['model'] not in TemperatureModel.__members__:
        valid = list(TemperatureModel.__members__)
        errors.append(f"[temperature] model='{config['temperature']['model']}' not in {valid}")

    if 'model' not in config['incident_angle']:
        errors.append("[incident_angle] missing required key 'model'")
    elif config['incident_angle']['model'] not in IncidentAngleModel.__members__:
        valid = list(IncidentAngleModel.__members__)
        errors.append(f"[incident_angle] model='{config['incident_angle']['model']}' not in {valid}")
    else:
        model = config['incident_angle']['model']
        for key in _IAM_GLASS_PARAMS[model]:
            if key not in config['incident_angle']:
                errors.append(f"[incident_angle] model='{model}' requires key '{key}'")

    cec_keys = ('alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s', 'Adjust')
    for key in cec_keys:
        if key not in config['module']:
            errors.append(f"[module] missing required CEC param '{key}'")

    for key in ('surface_tilt', 'surface_azimuth'):
        if key not in config['mounting']:
            errors.append(f"[mounting] missing required key '{key}'")

    registered_types = {cls.type for cls in FAILURE_CLASSES}
    all_names: list[str] = []
    for type_key, instances in config.get('effects', {}).items():
        if type_key not in registered_types:
            errors.append(
                f"[[effects.{type_key}]] no registered failure class has type='{type_key}' "
                f"(registered: {sorted(registered_types)})"
            )
            continue
        if not isinstance(instances, list):
            errors.append(
                f"[effects.{type_key}] must be declared as an array of tables ([[effects.{type_key}]])"
            )
            continue
        for i, instance in enumerate(instances):
            if 'name' not in instance:
                errors.append(f"[[effects.{type_key}]] entry #{i} missing required key 'name'")
            else:
                all_names.append(instance['name'])

    for duplicate in sorted({n for n in all_names if all_names.count(n) > 1}):
        errors.append(f"effect name '{duplicate}' used in multiple [[effects.*]] entries")

    if errors:
        raise ValueError(_format_errors(errors))

def _format_errors(errors: list[str]) -> str:
    """
    Formats a list of validation messages into one multi-line ``ValueError`` body.

    :param errors: list of individual violation strings.
    :return: a single newline-bulleted string suitable as the ``ValueError`` message.
    """

    joined = '\n  - '.join(errors)
    return f"Config file has {len(errors)} violation(s):\n  - {joined}"

def entry_point() -> None:
    parser = argparse.ArgumentParser(description="SOLARIS solar module simulation pipeline.")
    parser.add_argument("--cfg", type=Path, metavar="DIR",
                        help="File path from which to read configuration for simulation.")
    args = parser.parse_args()

    if not args.cfg or not Path(args.cfg).is_file():
        parser.error("Config file not specified.")

    with open(args.cfg, "rb") as f:
        cfg = tomllib.load(f)
        validate_config_file(cfg)
        orchestrator(cfg)

if __name__ == "__main__":
    entry_point()