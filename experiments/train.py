import difflib
import importlib
import os
import time
import uuid

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch as th
from stable_baselines3.common.utils import set_random_seed


from energy_net.env.single_entity_v0 import gym_env
from tests.single_agent_config import single_agent_cfgs

from energy_net.env.EnergyNetEnv import EnergyNetEnv
from tests.test_network import default_pcsunit, default_reward_function
from energy_net.network import Network
from energy_net.stratigic_entity import StrategicEntity

from stable_baselines3.common.env_checker import check_env

# Register custom envs
import rl_zoo3.import_envs  # noqa: F401
from exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS


def simulation_reward_function(state, action, new_state):
    grid_electricity = max(action.item() - state.get_consumption() + state.get_production(), 0)
    price = grid_electricity
    return -1 * price * grid_electricity

def train(
    algo="sac",
    tensorboard_log="./tmp/stable-baselines/",
    trained_agent="",
    truncate_last_trajectory=True,
    n_timesteps=-1,
    num_threads=-1,
    log_interval=-1,
    eval_freq=10000,
    optimization_log_path=None,
    eval_episodes=10,
    n_eval_envs=1,
    save_freq=-1,
    save_replay_buffer=False,
    log_folder="logs",
    seed=-1,
    vec_env="dummy",
    device="auto",
    n_trials=500,
    max_total_trials=None,
    optimize_hyperparameters=False,
    no_optim_plots=False,
    n_jobs=1,
    sampler="tpe",
    pruner="median",
    n_startup_trials=10,
    n_evaluations=None,
    storage=None,
    study_name=None,
    verbose=1,
    gym_packages=[],
    env_kwargs={},
    eval_env_kwargs={},
    hyperparams={},
    conf_file=None,
    unique_id=False,
    track=False,
    wandb_project_name="sb3",
    wandb_entity=None,
    show_progress=False,
    wandb_tags=[],
    env=None
) -> None:

    assert env is not None, "Environment must be provided"
    
    env_id = "energy_net-v0"

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f"_{uuid.uuid4()}" if unique_id else ""
    if seed < 0:
        # Seed but with a random one
        seed = np.random.randint(2**32 - 1, dtype="int64").item()  # type: ignore[attr-defined]

    set_random_seed(seed)

    # Setting num threads to 1 makes things run faster on cpu
    if num_threads > 0:
        if verbose > 1:
            print(f"Setting torch.num_threads to {num_threads}")
        th.set_num_threads(num_threads)

    if trained_agent != "":
        assert trained_agent.endswith(".zip") and os.path.isfile(trained_agent), "The trained_agent must be a valid path to a .zip file"

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {seed}")

    if track:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
            ) from e

        run_name = f"{env_id}__{algo}__{seed}__{int(time.time())}"
        tags = [*wandb_tags, f"v{sb3.__version__}"]
        run = wandb.init(
            name=run_name,
            project=wandb_project_name,
            entity=wandb_entity,
            tags=tags,
            config=vars(),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        tensorboard_log = f"runs/{run_name}"

    exp_manager = ExperimentManager(
        env=env,
        algo=algo,
        env_id=env_id,
        log_folder=log_folder,
        tensorboard_log=tensorboard_log,
        n_timesteps=n_timesteps,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        save_freq=save_freq,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
        eval_env_kwargs=eval_env_kwargs,
        trained_agent=trained_agent,
        optimize_hyperparameters=optimize_hyperparameters,
        storage=storage,
        study_name=study_name,
        n_trials=n_trials,
        max_total_trials=max_total_trials,
        n_jobs=n_jobs,
        sampler=sampler,
        pruner=pruner,
        optimization_log_path=optimization_log_path,
        n_startup_trials=n_startup_trials,
        n_evaluations=n_evaluations,
        truncate_last_trajectory=truncate_last_trajectory,
        uuid_str=uuid_str,
        seed=seed,
        log_interval=log_interval,
        save_replay_buffer=save_replay_buffer,
        verbose=verbose,
        vec_env_type=vec_env,
        n_eval_envs=n_eval_envs,
        no_optim_plots=no_optim_plots,
        device=device,
        config=conf_file,
        show_progress=show_progress,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        if track:
            # we need to save the loaded hyperparameters
            saved_hyperparams = saved_hyperparams
            assert run is not None  # make mypy happy
            run.config.setdefaults(vars())

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()

