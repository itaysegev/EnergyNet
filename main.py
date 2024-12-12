import gymnasium as gym
import energy_net.env
import os
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

def main():
    """
    Main function to interact with the EnergyNetEnv.
    Runs a simulation loop with random actions using gym.make for environment instantiation.
    """
    # Define configuration paths (update paths as necessary)
    env_config_path = 'configs/environment_config.yaml'
    iso_config_path = 'configs/iso_config.yaml'
    pcs_unit_config_path = 'configs/pcs_unit_config.yaml'
    log_file = 'logs/environment.log'
    id = 'MarketPlayerEnv-v0'
    # Attempt to create the environment using gym.make
    try:
        env = gym.make(
            id,
            disable_env_checker = True,
            env_config_path=env_config_path,
            iso_config_path=iso_config_path,
            pcs_unit_config_path=pcs_unit_config_path,
            log_file=log_file
        )
    except gym.error.UnregisteredEnv:
        print("Error: The environment 'MarketPlayerEnv-v0' is not registered. Please check your registration.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while creating the environment: {e}")
        return

    # Reset the environment to obtain the initial observation and info
    observation, info = env.reset()

    done = False
    truncated = False

    print("Starting MarketPlayer Simulation...")

    while not done and not truncated:
        # Sample a random action from the action space
        action = env.action_space.sample()
        
        # Take a step in the environment using the sampled action
        observation, reward, done, truncated, info = env.step(action)
        
        # Render the current state (if implemented)
        try:
            env.render()
        except NotImplementedError:
            pass  # Render not implemented; skip
        
        # Print observation, reward, and additional info
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        print("-" * 50)

    print("Simulation completed.")

    # Close the environment to perform any necessary cleanup
    env.close()
    
    
    
def train_and_evaluate_ppo(
    env_id='MarketPlayerEnv-v0',
    total_timesteps=1000,
    eval_episodes=10,
    log_dir='logs/ppo_energy_net_env',
    model_save_path='models/ppo_energy_net_env/ppo_energy_net_env',
    seed=42
):
    """
    Trains a PPO agent on the specified Gymnasium environment, evaluates its performance,
    and plots relevant metrics to visualize learning progress.

    Args:
        env_id (str): Gymnasium environment ID.
        total_timesteps (int): Total number of training timesteps.
        eval_episodes (int): Number of episodes to run during evaluation.
        log_dir (str): Directory to save training logs and plots.
        model_save_path (str): Path (without extension) to save the trained PPO model.
        seed (int): Random seed for reproducibility.
    """
    # Ensure directories exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Create and wrap the training environment with Monitor for logging
    train_env = gym.make(env_id)
    train_env = Monitor(train_env, filename=os.path.join(log_dir, 'train_monitor.csv'), allow_early_resets=True)
    train_env.reset(seed=seed)
    train_env.action_space.seed(seed)
    train_env.observation_space.seed(seed)

    # Create and wrap the evaluation environment with Monitor
    eval_env = gym.make(env_id)
    eval_env = Monitor(eval_env, filename=os.path.join(log_dir, 'eval_monitor.csv'), allow_early_resets=True)
    eval_env.reset(seed=seed+1)  # Different seed for evaluation
    eval_env.action_space.seed(seed+1)
    eval_env.observation_space.seed(seed+1)

    # Initialize PPO agent with MlpPolicy
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=1,
        tensorboard_log=log_dir,  # Logs for TensorBoard
        seed=seed,
    )

    class RewardCallback(BaseCallback):
        """
        Custom callback for recording episode rewards during training.
        """
        def __init__(self, verbose=0):
            super(RewardCallback, self).__init__(verbose)
            self.rewards = []

        def _on_step(self) -> bool:
            # Check if an episode has finished
            for info in self.locals.get('infos', []):
                if 'episode' in info.keys():
                    self.rewards.append(info['episode']['r'])
            return True

    reward_callback = RewardCallback()

    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=reward_callback, progress_bar=True)
    print("Training completed.")

    

    # Close training environment
    train_env.close()
    
    # Evaluate the trained agent
    print(f"Evaluating the trained agent over {eval_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=eval_episodes,
        deterministic=True
    )
    print(f"Mean Reward: {mean_reward} +/- {std_reward}")

    # Close evaluation environment
    eval_env.close()

    
    # Plot Training Rewards
    if reward_callback.rewards:
        plt.figure(figsize=(12, 6))
        plt.plot(reward_callback.rewards, label='Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards over Episodes')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, 'training_rewards.png'))
        plt.show()
    else:
        print("No training rewards recorded.")

    # Plot Evaluation Rewards
    eval_log_path = os.path.join(log_dir, 'eval_monitor.csv')
    if os.path.exists(eval_log_path):
        eval_data = pd.read_csv(eval_log_path)
        if 'r' in eval_data.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(eval_data['r'], marker='o', linestyle='-', label='Evaluation Episode Reward')
            plt.xlabel('Evaluation Episode')
            plt.ylabel('Reward')
            plt.title('Evaluation Rewards over Episodes')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(log_dir, 'evaluation_rewards.png'))
            plt.show()
        else:
            print("No evaluation rewards found in the evaluation monitor.")
    # else:
        print("Evaluation monitor CSV not found.")

    print("Training and evaluation process completed.")
    # Saving the Model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}.zip")
    
    
if __name__ == "__main__":
    main()
    # train_and_evaluate_ppo()