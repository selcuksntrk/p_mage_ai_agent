import configparser
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIKeyError(Exception):
    """Custom exception for API key related errors"""
    pass


def load_config(config_path='src/config.ini'):
    """
    Load configuration from config.ini file
    """
    try:
        config = configparser.ConfigParser()
        if Path(config_path).exists():
            config.read(config_path)
            return config
        logger.debug(f"Config file not found at {config_path}")
        return None
    except Exception as e:
        logger.warning(f"Error reading config file: {str(e)}")
        return None


def get_api_key(model_type, env_var_mapping):
    """
    Get API key for the specified model type, checking both environment variables
    and config.ini file.
    """
    env_var_name = env_var_mapping.get(model_type)
    if not env_var_name:
        raise APIKeyError(f"Unknown model type: {model_type}")

    # Try environment variable first
    api_key = os.environ.get(env_var_name)
    if api_key:
        logger.debug(f"API key found in environment variables for {model_type}")
        return api_key

    # Try config.ini if environment variable not found
    try:
        config = load_config()
        if config and 'API_KEYS' in config:
            api_key = config['API_KEYS'].get(env_var_name)
            if api_key:
                logger.debug(f"API key found in config.ini for {model_type}")
                return api_key
    except Exception as e:
        logger.error(f"Error reading config file: {str(e)}")

    # If we get here, no API key was found
    raise APIKeyError(
        f"API key not found for {model_type}. Please set the {env_var_name} "
        "environment variable or add it to config.ini under [API_KEYS] section."
    )


def check_api_key(model_type, env_var_mapping):
    """
    Check if API key exists for the specified model
    """
    try:
        get_api_key(model_type, env_var_mapping)
        return True, None
    except APIKeyError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error checking API key: {str(e)}"


# Example config.ini structure:
"""
[API_KEYS]
OPENAI_API_KEY = your_api_key_here
"""