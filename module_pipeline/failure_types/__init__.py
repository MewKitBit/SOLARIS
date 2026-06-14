from .abstract_baseline_failure import AbstractBaselineFailure
from .simple_soiling import SimpleSoilingModel

FAILURE_CLASSES: tuple[type[AbstractBaselineFailure], ...] = (SimpleSoilingModel,)
"""
Registry of concrete failure classes available to ``main.py``.

``main.py`` walks this tuple at config-load time and, for each class, looks up
``[[effects.<cls.type>]]`` in the TOML to instantiate every registered instance of
that class. When adding new failure class, include it here.
"""
