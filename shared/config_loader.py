import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(name):
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
