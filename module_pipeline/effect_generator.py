import numpy as np
from numpy.random import SeedSequence
from numpy.typing import NDArray

from .failure_types.abstract_baseline_failure import AbstractBaselineFailure


# Per-axis composition rule applied by ``compute_modifiers`` when overlapping effects touch the same axis.
AGGREGATION_RULES: dict[str, str] = {
    'G':    'multiplicative',
    'I_L':  'multiplicative',
    'R_sh': 'multiplicative',
    'T':    'additive',
    'R_s':  'additive',
    'I_0':  'additive',
}
VALID_PARAMS: tuple[str, ...] = tuple(AGGREGATION_RULES.keys())
_effects: tuple[AbstractBaselineFailure, ...] = ()
_master_seed: int = 0
_num_timesteps: int = 0


def configure(effects, master_seed: int, num_timesteps: int) -> None:
    """
    Configures the module state consumed by ``compute_modifiers`` and ``get_effect_names``.

    Called once at startup before the per-panel loop runs. Under multiprocessing, the
    configuration must be re-applied in each worker via a pool initializer, because
    spawn-method workers re-import the module with empty state.

    :param effects: iterable of effect instances. Each effect holds its own ``env_params``
                    reference and pre-computes any acceleration / hazard arrays in its
                    constructor, so per-panel calls do not re-traverse the environmental
                    timeseries.
    :param master_seed: root entropy for per-panel-per-effect RNG seeding.
    :param num_timesteps: length of the simulation time axis. Defines the shape of every
                          full-length modifier array returned by ``compute_modifiers``.
    """

    global _effects, _master_seed, _num_timesteps
    _effects = tuple(effects)
    _master_seed = master_seed
    _num_timesteps = num_timesteps


def validate_effects() -> None:
    """
    Validates that every registered effect declares its ``affected_columns`` (using only
    axes listed in ``VALID_PARAMS``).

    Reads each effect's class-level ``affected_columns`` attribute directly; no
    ``compute_progression`` call is involved. The runtime composition path in
    ``compute_modifiers`` trusts the declared columns without re-checking, so this
    startup pass is the only line of defense against typoed or invalid declarations.

    Collects every offending entry across all registered effects and raises a single
    ``ValueError`` listing them together, so a researcher sees all issues at once rather
    than fixing one and re-running to find the next.

    :raises ValueError: when any registered effect declares no ``affected_columns``,
                        declares a column not in ``VALID_PARAMS``, or shares a ``name``
                        with another registered effect.
    """

    errors: list[str] = []

    for effect in _effects:
        if not effect.affected_columns:
            errors.append(f"'{effect.name}' declares no affected_columns")
        else:
            for column in effect.affected_columns:
                if column not in VALID_PARAMS:
                    errors.append(f"'{effect.name}' declares invalid column '{column}'")

    if errors:
        joined = '; '.join(errors)
        raise ValueError(
            f"Effect contract violation(s): {joined}. Valid columns: {VALID_PARAMS}."
        )


def get_effect_names() -> tuple[str, ...]:
    """
    Returns the ordered tuple of registered effect names.

    Order matches the iterable passed to ``configure``. Used by ``data_generator`` to lay
    out per-effect attribution columns in a stable order across panels and chunks.

    :return: tuple of effect ``name`` strings.
    """

    return tuple(effect.name for effect in _effects)


def compute_modifiers(panel_num: int) -> tuple[dict[str, NDArray[np.float64]], dict[str, int]]:
    """
    Returns per-panel composed multiplicative modifiers and per-effect onset steps.

    For each registered effect, samples the onset step from a hierarchical RNG seeded on
    ``(master_seed, panel_num, effect_index)``. If onset falls within the simulated
    horizon, pulls the per-effect progression DataFrame from ``compute_progression`` and
    composes its column values onto a per-axis running array, dispatching by axis on
    ``AGGREGATION_RULES``:

    - **Multiplicative axes** (``G``, ``I_L``, ``R_sh``): progression values are
      fractional reductions ``r in [0, 1]``. The running array starts at the
      multiplicative identity ``1.0`` and accumulates as ``Pi(1 - r_i)`` across
      overlapping effects. The consumer applies it as ``baseline * factor``.
    - **Additive axes** (``T``, ``R_s``, ``I_0``): progression values are physical-
      quantity deltas (degrees C for ``T``, ohms for ``R_s``, amps for ``I_0``). The
      running array starts at the additive identity ``0.0`` and accumulates as
      ``Sigma delta_i``. The consumer applies it as ``baseline + delta``.

    Modifier dict keys are a subset of ``VALID_PARAMS`` covering the axes that
    registered effects actually touched on this panel. Axes that no effect touched are
    absent from the dict entirely, so the consumer's ``.get(key, 1.0)`` (or
    ``.get(key, 0.0)`` for additive axes) falls through to a free scalar broadcast.

    Onset dict maps effect ``name`` to the onset step index for effects that did trigger
    on this panel; effects that never triggered are absent.

    :param panel_num: zero-based panel index within the chunk.
    :return: tuple ``(modifiers, onsets)``.
    """

    modifiers: dict[str, NDArray[np.float64]] = {}
    onsets: dict[str, int] = {}

    for effect_index, effect in enumerate(_effects):
        seed = _derive_seed(panel_num, effect_index)
        onset = effect.compute_time_to_onset(seed)

        if onset < _num_timesteps:
            progression = effect.compute_progression(seed, onset, _num_timesteps)
            onsets[effect.name] = onset

            for column in progression.columns:
                segment = progression[column].to_numpy()
                rule = AGGREGATION_RULES[column]

                if rule == 'multiplicative':
                    if column not in modifiers:
                        modifiers[column] = np.ones(_num_timesteps, dtype=np.float64)
                    modifiers[column][onset:_num_timesteps] *= (1.0 - segment)
                else:
                    if column not in modifiers:
                        modifiers[column] = np.zeros(_num_timesteps, dtype=np.float64)
                    modifiers[column][onset:_num_timesteps] += segment

    return modifiers, onsets


def _derive_seed(panel_num: int, effect_index: int) -> int:
    """
    Derives an integer seed for one ``(panel, effect)`` pair from the master seed.

    Uses ``np.random.SeedSequence`` with a hierarchical spawn key so that the RNG streams
    for distinct ``(panel_num, effect_index)`` pairs are statistically independent.

    :param panel_num: zero-based panel index within the chunk.
    :param effect_index: zero-based index of the effect within the registered tuple.
    :return: 32-bit unsigned integer suitable as input to ``numpy.random.default_rng``.
    """

    seed_sequence = SeedSequence(entropy=_master_seed, spawn_key=(panel_num, effect_index))
    return int(seed_sequence.generate_state(1)[0])
