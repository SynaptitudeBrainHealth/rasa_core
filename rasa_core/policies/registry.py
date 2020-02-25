# Import all policies at one place to be able to to resolve them via a common module
# path. Don't do this in `__init__.py` to avoid importing them without need.

# noinspection PyUnresolvedReferences
from rasa_core.policies.embedding_policy import EmbeddingPolicy

# noinspection PyUnresolvedReferences
from rasa_core.policies.fallback import FallbackPolicy

# noinspection PyUnresolvedReferences
from rasa_core.policies.keras_policy import KerasPolicy

# noinspection PyUnresolvedReferences
from rasa_core.policies.memoization import MemoizationPolicy, AugmentedMemoizationPolicy

# noinspection PyUnresolvedReferences
from rasa_core.policies.sklearn_policy import SklearnPolicy

# noinspection PyUnresolvedReferences
from rasa_core.policies.form_policy import FormPolicy

# noinspection PyUnresolvedReferences
from rasa_core.policies.two_stage_fallback import TwoStageFallbackPolicy

# noinspection PyUnresolvedReferences
from rasa_core.policies.mapping_policy import MappingPolicy
