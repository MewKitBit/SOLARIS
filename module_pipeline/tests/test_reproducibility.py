"""
Reproducibility tests for the pipeline's RNG seeding and effect composition.

These cover the determinism guarantees that effect_generator provides to the rest of the
pipeline:
    - Per pair (panel, effect) seeds are a pure function of the master seed and the global panel id
    - The composed modifiers a panel receives are identical no matter how the fleet is partitioned

A throwaway in-test effect stands in for a real failure so the properties under test are not a failure model's .
"""

import numpy as np
from numpy.random import default_rng
from pandas import DataFrame, date_range

from .. import effect_generator
from ..failure_types.abstract_baseline_failure import AbstractBaselineFailure


class _StochasticGEffect(AbstractBaselineFailure):
    """
    Minimal effect whose onset and progression are derived purely from the supplied seed.

    Exists only to drive effect_generator's seeding and composition paths with reproducible,
    seed-varying output. It always triggers so every panel gets a populated modifier, and it
    touches the multiplicative G axis with fractional reductions.
    """

    type: str = "test_g"
    affected_columns: tuple[str, ...] = ('G',)

    def compute_time_to_onset(self, seed: int) -> int:
        rng = default_rng(seed)
        return int(rng.integers(0, len(self.env_params.index)))

    def compute_progression(self, seed: int, start_step: int, end_step: int) -> DataFrame:
        rng = default_rng(seed)
        reductions = rng.random(end_step - start_step)
        return DataFrame({'G': reductions}, index=self.env_params.index[start_step:end_step])


def _make_env(num_timesteps: int) -> DataFrame:
    """Builds an env frame of the right length since only its index is consumed by the test effect."""
    index = date_range('2020-01-01', periods=num_timesteps, freq='h', tz='UTC')
    return DataFrame(index=index)


def _assert_same_result(a, b) -> None:
    """Asserts two compute_modifiers returns are bit-identical in onsets and modifier arrays."""
    modifiers_a, onsets_a = a
    modifiers_b, onsets_b = b
    assert onsets_a == onsets_b
    assert modifiers_a.keys() == modifiers_b.keys()
    for axis in modifiers_a:
        assert np.array_equal(modifiers_a[axis], modifiers_b[axis])


def test_derive_seed_is_deterministic():
    """Same (master_seed, panel, effect_index) yields the same derived seed every call."""
    effect_generator.configure([], master_seed=42, num_timesteps=10)
    assert effect_generator._derive_seed(3, 1) == effect_generator._derive_seed(3, 1)


def test_derive_seed_separates_panel_and_effect():
    """
    The seed key keeps panel and effect distinct: neither coordinate is dropped, and the two are
    not collapsed into a commutative combination. Check our keying does not conflate
    distinct (panel, effect_index) pairs.
    """
    effect_generator.configure([], master_seed=42, num_timesteps=10)
    base = effect_generator._derive_seed(3, 1)
    assert effect_generator._derive_seed(4, 1) != base  # panel varies
    assert effect_generator._derive_seed(3, 2) != base  # effect varies
    assert effect_generator._derive_seed(1, 3) != base  # order matters (key is not commutative)


def test_derive_seed_depends_on_master_seed():
    """Changing the master seed changes the derived seed for the same (panel, effect) pair."""
    effect_generator.configure([], master_seed=1, num_timesteps=10)
    first = effect_generator._derive_seed(3, 1)
    effect_generator.configure([], master_seed=2, num_timesteps=10)
    second = effect_generator._derive_seed(3, 1)
    assert first != second


def test_compute_modifiers_is_reproducible():
    """Recomputing a panel's modifiers yields bit-identical onsets and arrays."""
    num_timesteps = 48
    effect = _StochasticGEffect('test', _make_env(num_timesteps))
    effect_generator.configure([effect], master_seed=42, num_timesteps=num_timesteps)
    _assert_same_result(effect_generator.compute_modifiers(7), effect_generator.compute_modifiers(7))


def test_distinct_panels_get_distinct_streams():
    """Different panels draw from independent streams, so their sampled onsets are not all identical."""
    num_timesteps = 48
    effect = _StochasticGEffect('test', _make_env(num_timesteps))
    effect_generator.configure([effect], master_seed=42, num_timesteps=num_timesteps)

    onsets = [effect_generator.compute_modifiers(panel)[1]['test'] for panel in range(50)]
    assert len(set(onsets)) > 1


def test_partitioning_does_not_change_modifiers():
    """
    A panel's modifiers depend only on its global id, not on how the fleet is split into chunks.

    Drives the same set of global panel ids through two different chunk partitionings (as
    main.py would) and asserts every panel gets bit-identical onsets and arrays either way, so
    changing chunk count or boundaries between runs cannot alter a single result.
    """
    num_timesteps = 48
    effect = _StochasticGEffect('test', _make_env(num_timesteps))
    effect_generator.configure([effect], master_seed=42, num_timesteps=num_timesteps)

    # Two partitionings of the same global ids [0, 20): different chunk counts and boundaries.
    partition_a = [range(0, 10), range(10, 20)]
    partition_b = [range(0, 4), range(4, 9), range(9, 20)]

    results_a = {p: effect_generator.compute_modifiers(p) for chunk in partition_a for p in chunk}
    results_b = {p: effect_generator.compute_modifiers(p) for chunk in partition_b for p in chunk}

    assert results_a.keys() == results_b.keys()
    for panel in results_a:
        _assert_same_result(results_a[panel], results_b[panel])
