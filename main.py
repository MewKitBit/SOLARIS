import argparse

from module_pipeline.enums import IncidentAngleModel, SingleDiodeMethod, TemperatureModel
from module_pipeline.failure_types import FAILURE_CLASSES
from pathlib import Path

# Workaround to tomllib not being included in Python versions under 3.11
try:
    import tomllib
except ImportError:
    import tomli as tomllib


def orchestrator(config: dict) -> None:
    # TODO: Load environmental and solar data from cfg pointed file
    # TODO: Configure effect generator by passing relevant data
    # TODO: Validate loaded effects from effect generator
    # TODO: Run data generator with chunking and parallelization as per cfg
    pass

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
                         'incident_angle', 'module')
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
        iam_required = {
            'PHYSICAL':    ('n', 'K', 'L'),
            'ASHRAE':      ('b',),
            'MARTIN_RUIZ': ('a_r',),
        }
        model = config['incident_angle']['model']
        for key in iam_required[model]:
            if key not in config['incident_angle']:
                errors.append(f"[incident_angle] model='{model}' requires key '{key}'")

    cec_keys = ('alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s', 'Adjust')
    for key in cec_keys:
        if key not in config['module']:
            errors.append(f"[module] missing required CEC param '{key}'")

    if 'mounting' not in config['module']:
        errors.append("[module.mounting] missing required subtable")
    else:
        for key in ('surface_tilt', 'surface_azimuth'):
            if key not in config['module']['mounting']:
                errors.append(f"[module.mounting] missing required key '{key}'")

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
                f"[effects.{type_key}] must be declared as an array of tables ([[effects.{type_key}]]), not a single table"
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