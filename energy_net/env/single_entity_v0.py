from . import energy_net_v0 as __energy_net_v0
from .wrappers.single_agent import SingleEntityWrapper as __SingleEntityWrapper
from gymnasium.wrappers import RescaleAction, PassiveEnvChecker, RecordEpisodeStatistics, TimeLimit


def gym_env(*args, **kwargs):
    """
    Create and configure a single entity environment for EnergyNet.

    This function initializes a parallel environment from the EnergyNet
    environment, wraps it with a single entity wrapper, sets the environment
    metadata, and rescales the action space to the range [-1, 1].

    Args:
        *args: Variable length argument list passed to the EnergyNet environment.
        **kwargs: Arbitrary keyword arguments passed to the EnergyNet environment.

    Returns:
        gym.Env: A configured single entity environment with rescaled action space.
    """
    energy_net_env = __energy_net_v0.parallel_env(*args, **kwargs)

    single_energy_net_env = __SingleEntityWrapper(energy_net_env)
    single_energy_net_env.unwrapped.metadata['name'] = 'single_entity_v0'
    single_energy_net_env = RescaleAction(single_energy_net_env, -1, 1)

    return single_energy_net_env





