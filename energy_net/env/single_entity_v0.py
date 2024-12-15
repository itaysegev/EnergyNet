
from __future__ import annotations


from typing import Optional, Tuple, Dict, Any, Union
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml
import logging

# from . import energy_net_v0 as __energy_net_v0
# from .wrappers.single_agent import SingleEntityWrapper as __SingleEntityWrapper
from energy_net.network import PCSUnitController

# def gym_env(*args, **kwargs):
#     """
#     Create and configure a single entity environment for EnergyNet.

#     This function initializes a parallel environment from the EnergyNet
#     environment, wraps it with a single entity wrapper, sets the environment
#     metadata, and rescales the action space to the range [-1, 1].

#     Args:
#         *args: Variable length argument list passed to the EnergyNet environment.
#         **kwargs: Arbitrary keyword arguments passed to the EnergyNet environment.

#     Returns:
#         gym.Env: A configured single entity environment with rescaled action space.
#     """
#     energy_net_env = __energy_net_v0.parallel_env(*args, **kwargs)

#     single_energy_net_env = __SingleEntityWrapper(energy_net_env)
#     single_energy_net_env.unwrapped.metadata['name'] = 'single_entity_v0'
#     # single_energy_net_env = RescaleAction(single_energy_net_env, -1, 1)

#     return single_energy_net_env



class MarketPlayerEnv(gym.Env):
    """
    A Gymnasium-compatible environment for simulating an energy network with
    battery storage, production, and consumption capabilities, managed by PCSUnit and ISO objects.

    Actions:
        Type: Box(1)
        Action                              Min                     Max
        Charging/Discharging Power           -max discharge rate     max charge rate

    Observation:
        Type: Box(4)
                                        Min                     Max
        Energy storage level (MWh)            0                       ENERGY_MAX
        Time (fraction of day)               0                       1
        Self Production (MWh)                0                       Inf
        Self Consumption (MWh)               0                       Inf
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        env_config_path: Optional[str] = 'configs/environment_config.yaml',
        iso_config_path: Optional[str] = 'configs/iso_config.yaml',
        pcs_unit_config_path: Optional[str] = 'configs/pcs_unit_config.yaml',
        log_file: Optional[str] = 'logs/environment.log',  # Path to the log file
        reward_type: str = 'cost'  # New parameter to specify the reward type
    ):
        """
        Constructs an instance of EnergyNetEnv.

        Args:
            render_mode: Optional rendering mode.
            env_config_path: Path to the environment YAML configuration file.
            iso_config_path: Path to the ISO YAML configuration file.
            pcs_unit_config_path: Path to the PCSUnit YAML configuration file.
            log_file: Path to the log file for environment logging.
            reward_type: Type of reward function to use.
        """
        super().__init__()  # Initialize the parent class
        self.controller = PCSUnitController(render_mode, env_config_path, iso_config_path, pcs_unit_config_path, log_file, reward_type)
        self.observation_space = self.controller.observation_space
        self.action_space = self.controller.action_space
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment to an initial state.

        Args:
            seed: Optional seed for random number generator.
            options: Optional settings like reward type.

        Returns:
            Tuple containing the initial observation and info dictionary.
        """
        super().reset(seed=seed)  # Reset the parent class's state
        return self.controller.reset(seed=seed, options=options)

    def step(self, action: Union[float, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes a single time step within the environment.

        Args:
        action (float or np.ndarray): Charging (+) or discharging (-) power.
            - If float: Represents the charging (+) or discharging (-) power directly.
            - If np.ndarray with shape (1,): The scalar value is extracted for processing.

        Returns:
            Tuple containing:
                - Next observation
                - Reward
                - Terminated flag
                - Truncated flag
                - Info dictionary
        """
        return self.controller.step(action)

    def _get_info(self) -> Dict[str, float]:
        """
        Provides additional information about the environment's state.

        Returns:
            Dict[str, float]: Dictionary containing the running average price.
        """
        return self.controller.get_info()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        return self.controller.load_config(config_path)

    def render(self, mode: Optional[str] = None):
        """
        Rendering method. Not implemented.

        Args:
            mode: Optional rendering mode.
        """
        self.controller.logger.warning("Render method is not implemented.")
        raise NotImplementedError("Rendering is not implemented.")

    def close(self):
        """
        Cleanup method. Closes loggers and releases resources.
        """
        self.controller.close()







