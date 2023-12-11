import json

def load_model_config(config_path):
    """Load model config

    Arguments
    ---------
    config_path : str
        Path of config

    Returns
    -------
    configs : dict, str
        Loaded config

    """
    with open(config_path, "r") as f:
        configs = json.load(f)
    return configs
