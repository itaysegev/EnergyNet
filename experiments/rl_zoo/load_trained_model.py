import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env

from tests.test_network import default_pcsunit, default_reward_function
from energy_net.network import Network
from energy_net.stratigic_entity import StrategicEntity
from energy_net.env.single_entity_v0 import gym_env

from energy_net.entities.pcsunit import PCSUnit
from energy_net.devices.params import StorageParams, ProductionParams, ConsumptionParams, DeviceParams
from energy_net.dynamics.consumption_dynamics.consumption_dynamics import GeneralLoad
from energy_net.dynamics.production_dynamics.production_dynamics import PVDynamics
from energy_net.dynamics.storage_dynamics.storage_dynamics import BatteryDynamics
from energy_net.dynamics.grid_dynamics import GridDynamics

from energy_net.devices.grid_device import GridDevice
from  energy_net.devices.storage_devices.local_storage import Battery
from  energy_net.devices.consumption_devices.local_consumer import ConsumerDevice
from  energy_net.devices.production_devices.local_producer import PrivateProducer

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import importlib
from copy import deepcopy
import os
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.utils import StoreDict, get_model_path


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



def test_pcsunit():
    # initialize consumer devices
        consumption_params_arr=[]
        file_name = 'first_day_data.xlsx'
        value_row_name = 'El [MWh]'
        time_row_name = 'Hour'
    
        general_load = GeneralLoad(file_name, value_row_name, time_row_name)
        consumption_params = ConsumptionParams(name='pcsunit_consumption', energy_dynamics=general_load, lifetime_constant=1, max_electric_power=general_load.max_electric_power, init_consum=general_load.init_power)
        consumption_params_arr.append(consumption_params)
        consumption_params_dict = {'pcsunit_consumption': consumption_params}
        
        # initialize storage devices
        storage_params_arr=[]
        storage_params = StorageParams(name = 'test_battery', energy_capacity = 4, power_capacity = 4,initial_charge = 0, charging_efficiency = 1,discharging_efficiency = 1, lifetime_constant = 1, energy_dynamics = BatteryDynamics())
        storage_params_arr.append(storage_params)
        storage_params_dict = {'test_battery': storage_params}

        # initialize production devices
        production_params_arr=[]
        value_row_name = 'Epv [MWh]'

        
        pv_dynamics = PVDynamics(file_name, value_row_name, time_row_name)

        production_params = ProductionParams(name='test_pv', max_production=pv_dynamics.max_production, efficiency=1, energy_dynamics=pv_dynamics, init_production = pv_dynamics.init_production)
        production_params_arr.append(production_params)
        production_params_dict = {'test_pv': production_params}
        
        
        
        grid_params = DeviceParams(name='grid', energy_dynamics=GridDynamics(), lifetime_constant=1)
        grid = GridDevice(grid_params)
        sub_entities= {name: ConsumerDevice(params) for name, params in consumption_params_dict.items()}
        sub_entities.update({name: Battery(params) for name, params in storage_params_dict.items()})
        sub_entities.update({name: PrivateProducer(params) for name, params in production_params_dict.items()})
        sub_entities.update({grid.name: grid})
        # initilaize pcsunit
        return PCSUnit(name="test_pcsuint", sub_entities=sub_entities, agg_func= None)


def simulation_reward_function(state, action, new_state):
    alpha = 0.01
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    
    if grid_electricity < 0:
        return -1_000
    price = grid_electricity
    return -1 * price * grid_electricity 

# Define the function to train and save models
def train_and_save_models(env, timesteps=1e6):
    # Train PPO
    model_ppo = PPO('MlpPolicy', env, verbose=1)
    model_ppo.learn(total_timesteps=timesteps)
    model_ppo.save("ppo_model")

    # Train SAC
    model_sac = SAC('MlpPolicy', env, verbose=1)
    model_sac.learn(total_timesteps=timesteps)
    model_sac.save("sac_model")

    # Train A2C
    model_a2c = A2C('MlpPolicy', env, verbose=1)
    model_a2c.learn(total_timesteps=timesteps)
    model_a2c.save("a2c_model")

# Define the function to run models on a new environment and save results
def collect_observations(env, model, num_steps=48):
    
    # Initialize lists to store results
    observations = []
   
    # Run models on the new environment
    obs = env.reset()
    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        observations.append(obs)
        
        if done:
            obs = env.reset()
            
    return observations



# Define your environment

# simulate the environment

# train_network =  Network(name="train_network", strategic_entities=[StrategicEntity(name="pcs_agent", network_entity=default_pcsunit(), reward_function=simulation_reward_function)])
# env = gym_env(network=train_network, simulation_start_time_step=0,
#                        simulation_end_time_step=48, episode_time_steps=48,
#                        seconds_per_time_step=60*30, initial_seed=0)

# simulate the environment

test_network =  Network(name="test_network", strategic_entities=[StrategicEntity(name="pcs_agent", network_entity=test_pcsunit(), reward_function=simulation_reward_function)])
    
test_env = gym_env(network=test_network, simulation_start_time_step=0,
                       simulation_end_time_step=48, episode_time_steps=48,
                       seconds_per_time_step=60*30, initial_seed=0)
env_name = "energy_net-v0"
_, model_path, log_path = get_model_path(
            2,
            'case2_logs',
            'ppo',
            env_name,
            True,
            None,
            False,
        )

stats_path = os.path.join(log_path, env_name)
hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)


env = create_test_env(
        test_env=test_env,
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=0,
        should_render=False,
        hyperparams=hyperparams,
    )

# # Train and save models
# train_and_save_models(env)

model_ppo = PPO.load("case2_logs/ppo/energy_net-v0_2/best_model.zip", env=env)
model_sac = SAC.load("case2_logs/sac/energy_net-v0_2/best_model.zip", env=env)
model_a2c = A2C.load("case2_logs/a2c/energy_net-v0_2/best_model.zip", env=env)

# Run models on a new environment and save results

ppo_observations = collect_observations(env, model_ppo)
sac_observations = collect_observations(env, model_sac)
a2c_observations = collect_observations(env, model_a2c)

# Extract the required columns from observations
data = {
    'Hour': [obs[1] for obs in ppo_observations],   # Assuming obs[1] is the Hour
    'Load': [obs[3] for obs in ppo_observations],   # Assuming obs[2] is the Load
    'PV': [obs[10] for obs in ppo_observations],     # Assuming obs[3] is the PV
    'PPO soc': [obs[4] for obs in ppo_observations],  # Assuming obs[4] is the soc
    'SAC soc': [obs[4] for obs in sac_observations],  # Assuming obs[4] is the soc
    'A2C soc': [obs[4] for obs in a2c_observations]   # Assuming obs[4] is the soc
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to an Excel file
output_file_path = 'model_comparisons.xlsx'
df.to_excel(output_file_path, index=False)

print(f"Results saved to {output_file_path}")