import argparse
import logging
import multiprocessing as mp
import time

from pandas import DataFrame, read_csv
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from module_pipeline import effect_generator
from module_pipeline.data_generator import generate_data
from module_pipeline.enums import IncidentAngleModel, SingleDiodeMethod, TemperatureModel
from module_pipeline.failure_types import FAILURE_CLASSES

# Workaround to tomllib not being included in Python versions under 3.11
try:
    import tomllib
except ImportError:
    import tomli as tomllib

logger = logging.getLogger("solaris")

# IAM glass parameters consumed per incident-angle model; the chosen model's set must be present.
_IAM_GLASS_PARAMS = {'PHYSICAL': ('n', 'K', 'L'), 'ASHRAE': ('b',), 'MARTIN_RUIZ': ('a_r',)}


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

    chunk_size = _compute_chunk_size(runtime['memory_cap_gb'], num_timesteps, runtime['num_workers'])
    chunks = _chunk_ranges(runtime['num_panels'], chunk_size)

    worker_args = (module_params, env_params, solar_positions, temp_model, iam_model, method,
                   observables_dir, attribution_dir)

    # Cap workers at the chunk count: more workers than chunks would just sit idle.
    num_workers = min(runtime['num_workers'], len(chunks))

    logger.info("Simulating %d panels over %d timesteps: %d chunk(s) of <=%d panels across %d worker(s)",
                runtime['num_panels'], num_timesteps, len(chunks), chunk_size, num_workers)
    logger.info("Effects: %s", ', '.join(effect_generator.get_effect_names()) or 'none')
    logger.info("Writing observables -> %s, attribution -> %s", observables_dir, attribution_dir)

    start = time.perf_counter()
    # logging_redirect_tqdm routes log records through tqdm.write so step logs don't garble the bar.
    with logging_redirect_tqdm(), tqdm(total=runtime['num_panels'], unit='panel', desc='Simulating') as bar:
        if num_workers == 1:
            for panel_ids in chunks:
                elapsed = _simulate_and_write(panel_ids, *worker_args)
                _record_chunk_done(bar, panel_ids, elapsed)
        else:
            # env_params is referenced both by the effects and standalone in worker_args, but it's the
            # same object in one pickle stream, so pickle's memo ships it once per worker (not twice).
            init_args = (effects, runtime['master_seed'], num_timesteps, *worker_args)
            # 'spawn' (not 'fork') so behaviour avoids fork+threaded-BLAS hazards
            # Spawned workers start with empty module state, which _init_worker re-establishes.
            with mp.get_context("spawn").Pool(num_workers, initializer=_init_worker, initargs=init_args) as pool:
                # imap_unordered yields each chunk's (range, elapsed) as it finishes, so the bar
                # advances on real progress instead of blocking until every chunk is done like map.
                for panel_ids, elapsed in pool.imap_unordered(_simulate_and_write_worker, chunks):
                    _record_chunk_done(bar, panel_ids, elapsed)

    logger.info("Done: %d panels in %d chunk(s) in %.1fs",
                runtime['num_panels'], len(chunks), time.perf_counter() - start)


def _record_chunk_done(bar: tqdm, panel_ids: range, elapsed: float) -> None:
    """
    Advances the overall progress bar by a finished chunk's panel count and logs its size and timing.

    :param bar: the ``tqdm`` bar tracking progress over the whole fleet.
    :param panel_ids: the global panel-id range of the chunk that just completed.
    :param elapsed: the chunk's wall time in seconds.
    """

    num_panels = len(panel_ids)
    bar.update(num_panels)
    logger.info("chunk done: %d panel(s) (%d..%d) in %.2fs",
                num_panels, panel_ids.start, panel_ids.stop, elapsed)


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


# Resident bytes per (panel, timestep) row of the observables frame, measured with
# DataFrame.memory_usage(deep=True): seven float64 I-V columns (56 B) plus the MultiIndex codes
# and 2 bytes of headroom (62 + 2)
_OBSERVABLE_ROW_BYTES = 64


def _compute_chunk_size(memory_cap_gb: float, num_timesteps: int, num_workers: int) -> int:
    """
    Derives how many panels fit in one chunk so all workers together stay within the cap.

    ``memory_cap_gb`` is the whole-process budget, split across ``num_workers`` before sizing: each
    worker holds one chunk-sized observables frame at a time, and up to ``num_workers`` of them are
    resident at once. The frame's footprint is its measured per-row cost (data plus MultiIndex, see
    ``_OBSERVABLE_ROW_BYTES``) times ``num_timesteps`` per panel; no headroom is reserved, so the
    cap should be set to the memory the operator will give the pipeline itself, leaving OS and
    hardware slack to them. Returns at least one panel so a tight cap still makes progress.

    :param memory_cap_gb: whole-process memory budget in gigabytes, from ``[runtime]``.
    :param num_timesteps: length of the simulation time axis.
    :param num_workers: number of worker processes the budget is divided across.
    :return: chunk size in panels (>= 1).
    """

    bytes_per_panel = num_timesteps * _OBSERVABLE_ROW_BYTES
    budget_bytes = (memory_cap_gb / num_workers) * 1e9
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


# Per-worker context for the parallel path. Under the 'spawn' start method each worker re-imports
# this module with empty state, so _init_worker re-establishes both the effect_generator config and
# the shared simulation inputs once per worker; tasks then carry only a panel-id range.
_WORKER_CONTEXT: dict = {}


def _simulate_and_write(panel_ids: range, module_params: dict, env_params: DataFrame,
                        solar_positions: DataFrame, temp_model: TemperatureModel,
                        iam_model: IncidentAngleModel, method: SingleDiodeMethod,
                        observables_dir: Path, attribution_dir: Path) -> float:
    """
    Simulates one chunk and writes its part-files. It's the unit of work shared by the sequential and
    parallel paths. All parameters beyond ``panel_ids`` are forwarded unchanged to ``generate_data``
    and ``_write_chunk``.

    Times itself so both paths report identical per-chunk wall time; in the parallel path this is the
    only place the duration is observable, since the parent sees a chunk only once it returns.

    :param panel_ids: the chunk's global panel-id range.
    :return: the chunk's wall time in seconds (simulate plus write).
    """

    start = time.perf_counter()
    observables, attribution = generate_data(
        module_params, env_params, solar_positions, temp_model, iam_model, panel_ids, method,
    )
    _write_chunk(observables_dir, attribution_dir, panel_ids, observables, attribution)
    return time.perf_counter() - start


def _init_worker(effects: list, master_seed: int, num_timesteps: int, *worker_args) -> None:
    """
    Pool initializer: runs once per worker to restore the state a spawned process lacks.

    Reconfigures ``effect_generator`` (its module-level state is empty after re-import) and stashes
    the shared simulation inputs for ``_simulate_and_write_worker`` to read.

    :param effects: effect instances to register, as built in the parent process.
    :param master_seed: root RNG seed.
    :param num_timesteps: simulation time-axis length.
    :param worker_args: the positional tail forwarded verbatim to ``_simulate_and_write``.
    """

    effect_generator.configure(effects, master_seed, num_timesteps)
    _WORKER_CONTEXT['args'] = worker_args


def _simulate_and_write_worker(panel_ids: range) -> tuple[range, float]:
    """
    Pool task body: simulates and writes one chunk using the per-worker context from
    ``_init_worker``. Takes only ``panel_ids`` so each task pickles cheaply.

    :param panel_ids: the chunk's global panel-id range.
    :return: ``(panel_ids, elapsed)`` so the parent can advance the bar and log the chunk's wall time.
    """

    elapsed = _simulate_and_write(panel_ids, *_WORKER_CONTEXT['args'])
    return panel_ids, elapsed


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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

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