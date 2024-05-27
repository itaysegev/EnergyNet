from abc import ABC, abstractmethod
import numpy as np

class NetworkAgent(ABC):
    """
    Abstract base class for network agents.
    """

    @abstractmethod
    def train(self, env, **kwargs):
        """
        Train the agents on the given environment.
        """
        pass

    @abstractmethod
    def eval(self, env, **kwargs):
        """
        Evaluate the agents on the given environment.
        """
        pass

    def get_action(self, observation=None, deterministic=False):
        """
        Get action to perform according to the current policy.
        """
        pass

class RandomAgent(NetworkAgent):
    """
    A random agents that takes random actions in the environment.
    """

    def __init__(self, env):
        self.env = env
        self.train_rewards = []
        self.eval_rewards = []
        self.soc = []
        self.action = []

    def train(self, total_timesteps=1000):
        pass

    def eval(self, n_episodes=100):
        observation = self.env.reset()[0]
        for _ in range(n_episodes):
            self.soc.append(observation[0])
            action = self.env.action_space.sample()
            self.action.append(action.item())
            observation, reward, done, _, info = self.env.step(action)
            self.eval_rewards.append(reward)


            if done:
                observation, info = self.env.reset()
        return np.mean(self.eval_rewards)
    
    def get_action(self, observation=None, deterministic=False):
        """
        Choose an action based on the given observation.

        Args:
            observation (np.ndarray): The observation from the environment.
            deterministic (bool): Whether to choose the action deterministically or stochastically.

        Returns:
            np.ndarray: The chosen action.
        """
        
        return self.env.action_space.sample()
                

