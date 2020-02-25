DEFAULT_SERVER_PORT = 5005

DEFAULT_SERVER_FORMAT = "http://localhost:{}"

DEFAULT_SERVER_URL = DEFAULT_SERVER_FORMAT.format(DEFAULT_SERVER_PORT)

MINIMUM_COMPATIBLE_VERSION = "0.14.0a3"

DOCS_BASE_URL = "https://rasa.com/docs/core"

DEFAULT_NLU_FALLBACK_THRESHOLD = 0.0

DEFAULT_CORE_FALLBACK_THRESHOLD = 0.0

DEFAULT_FALLBACK_ACTION = "action_default_fallback"

DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 minutes

REQUESTED_SLOT = 'requested_slot'

# start of special user message section
INTENT_MESSAGE_PREFIX = "/"

USER_INTENT_RESTART = INTENT_MESSAGE_PREFIX + "restart"

USER_INTENT_OUT_OF_SCOPE = 'out_of_scope'

# the lowest priority intended to be used by machine learning policies
DEFAULT_POLICY_PRIORITY = 1
# the priority intended to be used by mapping policies
MAPPING_POLICY_PRIORITY = 2
# the priority intended to be used by memoization policies
# it is higher than default and mapping to prioritize training stories
MEMOIZATION_POLICY_PRIORITY = 3
# the priority intended to be used by fallback policies
# it is higher than memoization to prioritize fallback
FALLBACK_POLICY_PRIORITY = 4
# the priority intended to be used by form policies
# it is the highest to prioritize form to the rest of the policies
FORM_POLICY_PRIORITY = 5