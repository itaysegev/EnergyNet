import importlib
import os
import sys
import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed
import matplotlib.pyplot as plt

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path

def eval(

    folder="logs/",
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
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print("Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            download_from_hub(
                algo=algo,
                env_name=env_name,
                exp_id=exp_id,
                folder=folder,
                organization="sb3",
                repo_name=None,
                force=False,
            )
            _, model_path, log_path = get_model_path(
                exp_id,
                folder,
                algo,
                env_name,
                load_best,
                load_checkpoint,
                load_last_checkpoint,
            )

    print(f"Loading {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    set_random_seed(seed)

    if num_threads > 0:
        if verbose > 1:
            print(f"Setting torch.num_threads to {num_threads}")
        th.set_num_threads(num_threads)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)

    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs.update(loaded_args["env_kwargs"])

    log_dir = reward_log if reward_log != "" else None

    env = create_test_env(
        env_name.gym_id,
        n_envs=n_envs,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=log_dir,
        should_render=not no_render,
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

# Example usage
if __name__ == "__main__":
    eval(
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
