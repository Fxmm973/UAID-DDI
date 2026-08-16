# Shared config loader: argparse defaults are read from configs/*.json
# (CLI arguments always override the file defaults).
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(name):
    """Load a config file (JSON/YAML) from the configs/ directory.
    Returns {} if the file is unavailable."""
    path = os.path.join(_ROOT, 'configs', name)
    if not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except ImportError:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    return data if isinstance(data, dict) else {}
