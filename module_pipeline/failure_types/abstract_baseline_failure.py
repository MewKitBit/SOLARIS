from abc import ABC, abstractmethod
from pandas import DataFrame

class AbstractBaselineFailure(ABC):
    """
    Abstract contract for failure modes.

    Subclasses encapsulate their own acceleration math, statistical onset distribution, and progression dynamics.
    The base class only declares the interface that ``effect_generator.compute_modifiers`` consumes.

    Subclasses must override ``affected_columns`` with a tuple of the column names their
    ``compute_progression`` returns. ``effect_generator.validate_effects`` reads this
    declaration at startup to verify only valid axes (members of ``VALID_PARAMS``) are
    claimed; the runtime composition path trusts the declaration without re-checking.

    :param name: short identifier used as the failure column group inside the per-panel DataFrame.
    :param env_params: ``pd.DataFrame`` of environmental conditions indexed by the simulation time axis.
    :param module_params: optional dict of static panel parameters that some subclasses may consult.
    """

    affected_columns: tuple[str, ...] = ()

    def __init__(self, name: str, env_params: DataFrame, module_params: dict | None = None):
        self.name = name
        self.env_params = env_params
        self.module_params = module_params

    @abstractmethod
    def compute_time_to_onset(self, seed: int) -> int:
        """
        Sample the first step at which this failure activates for one panel.

        :param seed: integer seed derived for this panel from the master RNG.
        :return: integer step index in ``[0, n_steps]``; a value at or above
                 ``n_steps`` means the failure never triggers within the simulated horizon.
        """
        pass

    @abstractmethod
    def compute_progression(self, seed: int, start_step: int, end_step: int) -> DataFrame:
        """
        Compute the per-step modifier progression from ``start_step`` to ``end_step``.

        The returned DataFrame is indexed by ``env_params.index[start_step:end_step]``
        and carries one to five columns whose names are a subset of ``IV_PARAMS``.
        Single-axis failures return a one-column DataFrame; multi-axis failures (e.g.
        microcracks affecting both ``R_s`` and ``I_L``) return additional columns.
        Modifier values are fractional relative to the baseline I-V parameters.

        :param seed: integer seed derived for this panel from the master RNG.
        :param start_step: inclusive step index at which the progression begins, as
                           returned by ``compute_time_to_onset``.
        :param end_step: exclusive step index marking the end of the simulated horizon.
        :return: ``pd.DataFrame`` whose columns form a non-empty subset of ``IV_PARAMS``.
        """
        pass