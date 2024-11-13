
import json
from pathlib import Path

from tests.test_network import default_network



ENV_CFG_FILE = Path(__file__).parent / 'test_env_configs.json'


def get_env_cfgs():
    with open(ENV_CFG_FILE, 'r') as f:
        env_cfgs = json.load(f)
    
    env_cfgs['single_entity_simple']['network'] = default_network()
    return env_cfgs


test_env_cfgs = get_env_cfgs()

single_agent_cfgs = {k: cfg for k, cfg in test_env_cfgs.items()}