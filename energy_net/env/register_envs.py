# energy_net/env/register_envs.py

from gymnasium.envs.registration import register

print("Registering MarketPlayerEnv-v0")
register(
    id='MarketPlayerEnv-v0',
    entry_point='energy_net.env.single_entity_v0:MarketPlayerEnv',
    # Optional parameters:
    # max_episode_steps=1000,
    # reward_threshold=100.0,
    # nondeterministic=False,
)
