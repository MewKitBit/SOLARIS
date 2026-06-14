import numpy as np
from numpy.random import default_rng
from pandas import DataFrame

from .abstract_baseline_failure import AbstractBaselineFailure

class SimpleSoilingModel(AbstractBaselineFailure):
    """
    Simple soiling: gradual dust and debris accumulation that dims the irradiance reaching the cells.

    Soiling is always-on (onset at the first step), since a deployed panel begins accumulating dust
    immediately. The per-panel soiling level starts at a random initial fraction (lognormal, so it is
    strictly positive and right-skewed: most panels start lightly soiled, a few heavily) and then
    evolves as a random walk: each step adds a draw from ``Normal(avg_hourly_progression,
    std_deviation)``, so the level drifts upward on average while occasional negative draws model
    partial cleaning by rain or wind. The running level is bounded to ``[0, max_soiling_effect]``.
    Single-axis: only the effective irradiance ``G`` is affected, as a fractional reduction applied
    multiplicatively by the composition layer (``baseline * (1 - r)``).

    :param env_params: ``pd.DataFrame`` of environmental conditions indexed by the simulation time
                       axis. This model uses only its index, to timestamp the progression.
    :param max_soiling_effect: asymptotic cap on the fractional irradiance loss, in ``[0, 1]``
                               (e.g. ``0.3`` caps the loss at 30%).
    :param avg_hourly_progression: mean per-step increase of the soiling fraction (positive drift).
    :param std_deviation: standard deviation of the per-step increment; negative draws model
                          partial cleaning.
    :param starting_median_fraction: median initial soiling level, as a fraction of
                                     ``max_soiling_effect``. Defaults to ``0.25``.
    :param starting_shape_sigma: sigma of the underlying normal of the lognormal initial-level draw;
                                 larger values widen the spread of starting levels. Defaults to ``0.5``.
    :param name: short identifier used as the attribution column prefix for this effect.
    """

    type: str = "soiling_variant"
    affected_columns: tuple[str, ...] = ('G',)

    def __init__(self, env_params: DataFrame, max_soiling_effect: float, avg_hourly_progression: float,
                 std_deviation: float, starting_median_fraction: float = 0.25,
                 starting_shape_sigma: float = 0.5, name: str = "simple-soiling"):
        super().__init__(name=name, env_params=env_params)
        if not 0.0 <= max_soiling_effect <= 1.0:
            raise ValueError(
                f"max_soiling_effect is a fractional irradiance reduction and must lie in [0, 1] "
                f"(got {max_soiling_effect})"
            )
        self.max_soiling_effect = max_soiling_effect
        self.avg_hourly_progression = avg_hourly_progression
        self.std_deviation = std_deviation
        self.starting_median_fraction = starting_median_fraction
        self.starting_shape_sigma = starting_shape_sigma


    def compute_time_to_onset(self, seed: int) -> int:
        """
        Panel is always soiling from the start, return first step as the start of the soiling effect.
        """
        return 0

    def compute_progression(self, seed: int, start_step: int, end_step: int) -> DataFrame:
        """
        Accumulate a per-step soiling fraction from a random starting level, capped at the maximum.

        The panel begins at a random initial soiling level (lognormal, so it is strictly positive and
        right-skewed: most panels start lightly soiled, a few heavily). From there the level performs
        a random walk with positive drift ``avg_hourly_progression`` and spread ``std_deviation`` per
        step, modelling gradual accumulation with occasional partial cleaning (negative draws). The
        running level is bounded to ``[0, max_soiling_effect]``.

        :return: single-column ``'G'`` frame of fractional irradiance reductions in
                 ``[0, max_soiling_effect]``; the consumer applies each as ``baseline * (1 - r)``.
        """
        # Seed a local generator so the progression is reproducible and independent per panel.
        # np.random.* would draw from the global generator and ignore `seed`, breaking determinism.
        rng = default_rng(seed)

        # Random starting soiling fraction. Lognormal for positivity and right skew; the underlying
        # normal's mean is set so the median start is `starting_median_fraction` of the cap.
        starting_soiling = rng.lognormal(
            mean=np.log(self.starting_median_fraction * self.max_soiling_effect),
            sigma=self.starting_shape_sigma,
        )

        # Per-step increments around the average hourly rate; negative draws model partial cleaning.
        soiling_per_step = rng.normal(loc=self.avg_hourly_progression,
                                      scale=self.std_deviation,
                                      size=end_step - start_step)

        # Accumulate from the starting level, then bound to the physical range.
        soiling_level = starting_soiling + np.cumsum(soiling_per_step)
        soiling_capped = np.clip(soiling_level, 0.0, self.max_soiling_effect)

        return DataFrame({'G': soiling_capped}, index=self.env_params.index[start_step:end_step])

