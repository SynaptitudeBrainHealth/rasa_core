import glob
import logging
import os
import tempfile
from typing import Text, Union, Optional, List, Dict, Tuple

from rasa.constants import (
    DEFAULT_MODELS_PATH
)

from rasa_core.exceptions import ModelNotFound
from rasa_core.utils import TempDirectoryPath


# Type alias for the fingerprint
Fingerprint = Dict[Text, Union[Text, List[Text], int, float]]

logger = logging.getLogger(__name__)

FINGERPRINT_FILE_PATH = "fingerprint.json"

FINGERPRINT_CONFIG_KEY = "config"
FINGERPRINT_CONFIG_CORE_KEY = "core-config"
FINGERPRINT_CONFIG_NLU_KEY = "nlu-config"
FINGERPRINT_DOMAIN_KEY = "domain"
FINGERPRINT_RASA_VERSION_KEY = "version"
FINGERPRINT_STORIES_KEY = "stories"
FINGERPRINT_NLU_DATA_KEY = "messages"
FINGERPRINT_TRAINED_AT_KEY = "trained_at"


def get_model(model_path: Text = DEFAULT_MODELS_PATH) -> TempDirectoryPath:
    """Gets a model and unpacks it. Raises a `ModelNotFound` exception if
    no model could be found at the provided path.

    Args:
        model_path: Path to the zipped model. If it's a directory, the latest
                    trained model is returned.

    Returns:
        Path to the unpacked model.

    """
    if not model_path:
        raise ModelNotFound("No path specified.")
    elif not os.path.exists(model_path):
        raise ModelNotFound("No file or directory at '{}'.".format(model_path))

    if os.path.isdir(model_path):
        model_path = get_latest_model(model_path)
        if not model_path:
            raise ModelNotFound(
                "Could not find any Rasa model files in '{}'.".format(model_path)
            )
    elif not model_path.endswith(".tar.gz"):
        raise ModelNotFound(
            "Path '{}' does not point to a Rasa model file.".format(model_path)
        )

    return unpack_model(model_path)


def get_latest_model(model_path: Text = DEFAULT_MODELS_PATH) -> Optional[Text]:
    """Gets the latest model from a path.

    Args:
        model_path: Path to a directory containing zipped models.

    Returns:
        Path to latest model in the given directory.

    """
    if not os.path.exists(model_path) or os.path.isfile(model_path):
        model_path = os.path.dirname(model_path)

    list_of_files = glob.glob(os.path.join(model_path, "*.tar.gz"))

    if len(list_of_files) == 0:
        return None

    return max(list_of_files, key=os.path.getctime)


def unpack_model(
    model_file: Text, working_directory: Optional[Text] = None
) -> TempDirectoryPath:
    """Unpacks a zipped Rasa model.

    Args:
        model_file: Path to zipped model.
        working_directory: Location where the model should be unpacked to.
                           If `None` a temporary directory will be created.

    Returns:
        Path to unpacked Rasa model.

    """
    import tarfile

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    tar = tarfile.open(model_file)

    # All files are in a subdirectory.
    tar.extractall(working_directory)
    tar.close()
    logger.debug("Extracted model to '{}'.".format(working_directory))

    return TempDirectoryPath(working_directory)


def get_model_subdirectories(
    unpacked_model_path: Text
) -> Tuple[Optional[Text], Optional[Text]]:
    """Returns paths for Core and NLU model directories, if they exist.
    If neither directories exist, a `ModelNotFound` exception is raised.

    Args:
        unpacked_model_path: Path to unpacked Rasa model.

    Returns:
        Tuple (path to Core subdirectory if it exists or `None` otherwise,
               path to NLU subdirectory if it exists or `None` otherwise).

    """
    core_path = os.path.join(unpacked_model_path, "core")

    if not os.path.isdir(core_path):
        core_path = None

    if not core_path:
        raise ModelNotFound(
            "No NLU or Core data for unpacked model at: '{}'.".format(
                unpacked_model_path
            )
        )

    return core_path
