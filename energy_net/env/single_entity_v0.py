from . import energy_net_v0 as __energy_net_v0
from .wrappers.single_agent import SingleEntityWrapper as __SingleEntityWrapper
from gymnasium.wrappers import RescaleAction, PassiveEnvChecker, RecordEpisodeStatistics, TimeLimit


def gym_env(*args, **kwargs):
    energy_net_env = __energy_net_v0.parallel_env(*args, **kwargs)

    single_energy_net_env = __SingleEntityWrapper(energy_net_env)
    single_energy_net_env.unwrapped.metadata['name'] = 'single_entity_v0'
    single_energy_net_env = RescaleAction(single_energy_net_env, -1, 1)

    return single_energy_net_env