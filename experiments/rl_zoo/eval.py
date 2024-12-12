import importlib
import os
import sys
import numpy as np
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, Optional
from copy import deepcopy
import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.utils import get_model_path


from energy_net.env.single_entity_v0 import gym_env

from energy_net.network import Network

from energy_net.components.pcsunit import PCSUnit
from energy_net.components.params import StorageParams, ProductionParams, ConsumptionParams, DeviceParams
from energy_net.dynamics.consumption_dynamics.consumption_dynamics import GeneralLoad
from energy_net.dynamics.production_dynamics.production_dynamics import PVDynamics
from energy_net.dynamics.storage_dynamics.storage_dynamics import BatteryDynamics
from energy_net.dynamics.grid_dynamics import GridDynamics

from energy_net.components.grid_device import GridDevice
from  energy_net.components.storage_devices.local_storage import Battery
from  energy_net.components.consumption_devices.local_consumer import ConsumerDevice
from  energy_net.components.production_devices.local_producer import PrivateProducer
from energy_net.config import DEFAULT_LIFETIME_CONSTANT
from energy_net.stratigic_entity import StrategicEntity

import gymnasium as gym
import stable_baselines3 as sb3  # noqa: F401
import torch as th  # noqa: F401
import yaml

from huggingface_sb3 import EnvironmentName

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize


def get_wrapper_class(hyperparams: Dict[str, Any], key: str = "env_wrapper") -> Optional[Callable[[gym.Env], gym.Env]]:
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    Works also for VecEnvWrapper with the key "vec_env_wrapper".

    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    for multiple, specify a list:

    env_wrapper:
        - rl_zoo3.wrappers.PlotActionWrapper
        - rl_zoo3.wrappers.TimeFeatureWrapper


    :param hyperparams:
    :return: maybe a callable to wrap the environment
        with one or multiple gym.Wrapper
    """

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if key in hyperparams.keys():
        wrapper_name = hyperparams.get(key)

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
            # Handle keyword arguments
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {wrapper_name}. "
                    "You should check the indentation."
                )
                wrapper_dict = wrapper_name
                wrapper_name = next(iter(wrapper_dict.keys()))
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env: gym.Env) -> gym.Env:
            """
            :param env:
            :return:
            """
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None

def create_test_env(
    test_env,
    n_envs: int = 1,
    stats_path: Optional[str] = None,
    seed: int = 0,
    log_dir: Optional[str] = None,
    should_render: bool = True,
    hyperparams: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create environment for testing a trained agent

    :param env_id:
    :param n_envs: number of processes
    :param stats_path: path to folder containing saved running averaged
    :param seed: Seed for random number generator
    :param log_dir: Where to log rewards
    :param should_render: For Pybullet env, display the GUI
    :param hyperparams: Additional hyperparams (ex: n_stack)
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :return:
    """
    # Create the environment and wrap it if necessary
    assert hyperparams is not None
    env_wrapper = get_wrapper_class(hyperparams)

    hyperparams = {} if hyperparams is None else hyperparams

    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    vec_env_kwargs: Dict[str, Any] = {}
    # Avoid potential shared memory issue
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

    # Fix for gym 0.26, to keep old behavior
    env_kwargs = env_kwargs or {}
    env_kwargs = deepcopy(env_kwargs)
    if "render_mode" not in env_kwargs and should_render:
        env_kwargs.update(render_mode="human")

    # spec = gym.spec(env_id)

    # Define make_env here, so it works with subprocesses
    # when the registry was modified with `--gym-packages`
    # See https://github.com/HumanCompatibleAI/imitation/pull/160
    def make_env(**kwargs) -> gym.Env:
        return test_env

    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        monitor_dir=log_dir,
        seed=seed,
        wrapper_class=env_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,  # type: ignore[arg-type]
        vec_env_kwargs=vec_env_kwargs,
    )

    if "vec_env_wrapper" in hyperparams.keys():
        vec_env_wrapper = get_wrapper_class(hyperparams, "vec_env_wrapper")
        assert vec_env_wrapper is not None
        env = vec_env_wrapper(env)  # type: ignore[assignment, arg-type]
        del hyperparams["vec_env_wrapper"]

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams["normalize"]:
            print("Loading running average")
            print(f"with params: {hyperparams['normalize_kwargs']}")
            path_ = os.path.join(stats_path, "vecnormalize.pkl")
            if os.path.exists(path_):
                env = VecNormalize.load(path_, env)
                # Deactivate training and reward normalization
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {path_} not found")

        n_stack = hyperparams.get("frame_stack", 0)
        if n_stack > 0:
            print(f"Stacking {n_stack} frames")
            env = VecFrameStack(env, n_stack)
    return env

def eval(
    test_env,
    folder,
    algo="ppo",
    n_timesteps=1000,
    num_threads=-1,
    n_envs=1,
    exp_id=0,
    verbose=1,
    no_render=False,
    deterministic=False,
    device="auto",
    load_best=True,
    load_checkpoint=None,
    load_last_checkpoint=False,
    stochastic=False,
    norm_reward=False,
    seed=0,
    reward_log="",
    gym_packages=None,
    env_kwargs=None,
    custom_objects=False,
    progress=False
):
    
    env = "energy_net-v0"
    env_name = EnvironmentName(env)
    try:
        _, model_path, log_path = get_model_path(
            exp_id,
            folder,
            algo,
            env_name,
            load_best,
            load_checkpoint,
            load_last_checkpoint,
        )
    except (AssertionError, ValueError) as e:
        raise e
        
    print(f"Loading {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    set_random_seed(seed)

    if num_threads > 0:
        if verbose > 1:
            print(f"Setting torch.num_threads to {num_threads}")
        th.set_num_threads(num_threads)

    stats_path = os.path.join(log_path, env_name)
    print(stats_path)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)

    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs.update(loaded_args["env_kwargs"])

    log_dir = reward_log if reward_log != "" else None

    env = create_test_env(
        test_env=test_env,
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=0,
        log_dir=log_dir,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=seed)
    if algo in off_policy_algos:
        kwargs.update(dict(buffer_size=1))
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects_dict = {}
    if newer_python_version or custom_objects:
        custom_objects_dict = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects_dict, device=device, **kwargs)
    obs = env.reset()

    deterministic = not stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    successes = []
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)

    generator = range(n_timesteps)
    if progress:
        if tqdm is None:
            raise ImportError("Please install tqdm and rich to use the progress bar")
        generator = tqdm(generator)

    try:
        for _ in generator:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            obs, reward, done, infos = env.step(action)

            episode_start = done

            if not no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if n_envs == 1:
                
                if done and verbose > 0:
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print("Episode Length", ep_len)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0

                if done and infos[0].get("is_success") is not None:
                    if verbose > 1:
                        print("Success?", infos[0].get("is_success", False))

                    if infos[0].get("is_success") is not None:
                        successes.append(infos[0].get("is_success", False))
                        episode_reward, ep_len = 0.0, 0

    except KeyboardInterrupt:
        pass

    if verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()

    # Plotting
    if len(episode_rewards) > 0:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.subplot(1, 2, 2)
        plt.plot(episode_lengths)
        plt.title("Episode Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Length")

        plt.tight_layout()
        plt.savefig("results.svg")
        plt.show()
        
        
def losses_simulation_reward_function(state, action, new_state):
    alpha = 0.01
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    
    if grid_electricity < 0:
        return -1 *1000
    price = grid_electricity
    price = price + alpha * price * price
    return -1 * price * price


def simulation_reward_function(state, action, new_state):
    alpha = 0.01
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    
    if grid_electricity < 0:
        return -1 *1000
    price = grid_electricity
    return -1 * price * grid_electricity 



def test_pcsunit():
    # initialize consumer components
        consumption_params_arr=[]
        file_name = 'first_day_data.xlsx'
        value_row_name = 'El [MWh]'
        time_row_name = 'Hour'
    
        general_load = GeneralLoad(file_name, value_row_name, time_row_name)
        consumption_params = ConsumptionParams(name='pcsunit_consumption', energy_dynamics=general_load, lifetime_constant=1, max_electric_power=general_load.max_electric_power)
        consumption_params_arr.append(consumption_params)
        consumption_params_dict = {'pcsunit_consumption': consumption_params}
        
        # initialize storage components
        storage_params_arr=[]
        storage_params = StorageParams(name = 'test_battery', energy_capacity = 4, power_capacity = 4,initial_charge = 0, charging_efficiency = 0.9,discharging_efficiency = 0.9, lifetime_constant = 1, energy_dynamics = BatteryDynamics())
        storage_params_arr.append(storage_params)
        storage_params_dict = {'test_battery': storage_params}

        # initialize production components
        production_params_arr=[]
        value_row_name = 'Epv [MWh]'

        
        pv_dynamics = PVDynamics(file_name, value_row_name, time_row_name)

        production_params = ProductionParams(name='test_pv', max_production=pv_dynamics.max_production, efficiency=1, energy_dynamics=pv_dynamics)
        production_params_arr.append(production_params)
        production_params_dict = {'test_pv': production_params}
        
        
        
        grid_params = DeviceParams(name='grid', energy_dynamics=GridDynamics(), lifetime_constant=DEFAULT_LIFETIME_CONSTANT)
        grid = GridDevice(grid_params)
        sub_entities= {name: ConsumerDevice(params) for name, params in consumption_params_dict.items()}
        sub_entities.update({name: Battery(params) for name, params in storage_params_dict.items()})
        sub_entities.update({name: PrivateProducer(params) for name, params in production_params_dict.items()})
        sub_entities.update({grid.name: grid})
        # initilaize pcsunit
        return PCSUnit(name="test_pcsuint", sub_entities=sub_entities, agg_func= None)


# Example usage
if __name__ == "__main__":
    
    strategic_entities = [StrategicEntity(name="pcs_agent", grid_entity=test_pcsunit(), reward_function=simulation_reward_function)]
    network = Network(name="test_network", strategic_entities=strategic_entities)

    env = gym_env(network=network, simulation_start_time_step=0,
                       simulation_end_time_step=48, episode_time_steps=48,
                       seconds_per_time_step=60*30)
    
    # try:
    #     check_env(env)
    #     print('Passed test!! EnergyNetEnv is compatible with SB3 when using the StableBaselines3Wrapper.')
    # finally:
    #     pass
    
    
    eval(
        test_env= env,
        folder="case2_logs",
        algo="ppo",
        n_timesteps=48,
        num_threads=1,
        n_envs=1,
        exp_id=0,
        verbose=1,
        no_render=False,
        deterministic=True,
        device="cpu",
        load_best=True,
        load_checkpoint=None,
        load_last_checkpoint=False,
        stochastic=False,
        norm_reward=False,
        seed=0,
        reward_log="",
        env_kwargs={},
        custom_objects=False,
        progress=False
    )
