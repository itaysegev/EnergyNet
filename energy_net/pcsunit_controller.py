from energy_net.components.grid_entity import GridEntity
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import os
from gymnasium import spaces
import yaml
import logging

from energy_net.components.pcsunit import PCSUnit
from energy_net.dynamics.energy_dynamcis import EnergyDynamics
from energy_net.dynamics.energy_dynamcis import ModelBasedDynamics
from energy_net.dynamics.production_dynamics.deterministic_production import DeterministicProduction
from energy_net.dynamics.consumption_dynamics.deterministic_consumption import DeterministicConsumption
from energy_net.dynamics.storage_dynamics.deterministic_battery import DeterministicBattery  # Import the new dynamics
from energy_net.dynamics.energy_dynamcis import DataDrivenDynamics
from energy_net.utils.iso_factory import iso_factory
from energy_net.utils.logger import setup_logger  # Import the logger setup


# Import all reward classes
from energy_net.rewards.base_reward import BaseReward
from energy_net.rewards.cost_reward import CostReward



class PCSUnitController():
    """
    Actions:
        Type: Box
            - If multi_action=False:
                Charging/Discharging Power: continuous scalar
            - If multi_action=True:
                [Charging/Discharging Power, Consumption Action, Production Action]

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

        # Set up logger
        self.logger = setup_logger('EnergyNetEnv', log_file)
        self.logger.info("Initializing EnergyNetEnv.")

        # Load configurations
        self.env_config: Dict[str, Any] = self.load_config(env_config_path)
        self.iso_config: Dict[str, Any] = self.load_config(iso_config_path)
        self.pcs_unit_config: Dict[str, Any] = self.load_config(pcs_unit_config_path)

        # Initialize ISO using the factory
        iso_type: str = self.iso_config.get('type', 'HourlyPricingISO')  # Default to HourlyPricingISO
        iso_parameters: Dict[str, Any] = self.iso_config.get('parameters', {})
        self.ISO = iso_factory(iso_type, iso_parameters)
        self.logger.info(f"Initialized ISO with type: {iso_type} and parameters: {iso_parameters}")
        
        # Initialize PCSUnit with dynamics and configuration
        self.PCSUnit: PCSUnit = PCSUnit(
            config=self.pcs_unit_config,
            log_file=log_file
        )
        
        self.logger.info("Initialized PCSUnit with all components.")

        # Define Action Space
        energy_config: Dict[str, Any] = self.pcs_unit_config['battery']['model_parameters']
        
        
        # Load action configurations
        self.multi_action: bool = self.pcs_unit_config.get('action', {}).get('multi_action', False)
        self.production_action_enabled: bool = self.pcs_unit_config.get('action', {}).get('production_action', {}).get('enabled', False)
        self.consumption_action_enabled: bool = self.pcs_unit_config.get('action', {}).get('consumption_action', {}).get('enabled', False)

        
        if self.multi_action:
            # Load production action limits
            self.production_action_min: float = self.pcs_unit_config.get('action', {}).get('production_action', {}).get('min')
            self.production_action_max: float = self.pcs_unit_config.get('action', {}).get('production_action', {}).get('max')
            
            # Load consumption action limits
            self.consumption_action_min: float = self.pcs_unit_config.get('action', {}).get('consumption_action', {}).get('min')
            self.consumption_action_max: float = self.pcs_unit_config.get('action', {}).get('consumption_action', {}).get('max')
            
            action_space_low = np.array([
                -energy_config['discharge_rate_max'],  # Charging/Discharging Power
                self.consumption_action_min,          # Consumption Action
                self.production_action_min            # Production Action
            ], dtype=np.float32)
            
            action_space_high = np.array([
                energy_config['charge_rate_max'],     # Charging/Discharging Power
                self.consumption_action_max,          # Consumption Action
                self.production_action_max            # Production Action
            ], dtype=np.float32)
            
            self.action_space: spaces.Box = spaces.Box(
                low=action_space_low,
                high=action_space_high,
                shape=(3,),
                dtype=np.float32
            )
            self.logger.info(f"Defined action space with additional actions: low={action_space_low}, high={action_space_high}")
        
        else:
            self.action_space: spaces.Box = spaces.Box(
                low=-energy_config['discharge_rate_max'], # specifies the maximum rate at which energy can be discharged from the energy storage system per unit time in the environment.
                high=energy_config['charge_rate_max'], # specifies the maximum rate at which energy can be charged into the energy storage system per unit time in the environment.
                shape=(1,),
                dtype=np.float32
            )
            self.logger.info(f"Defined action space: low={-energy_config['discharge_rate_max']}, high={energy_config['charge_rate_max']}")

        # Define Observation Space
        self.observation_space: spaces.Box = spaces.Box(
            low=np.array([
                energy_config['min'],
                0.0,
                0.0,
                0.0
            ], dtype=np.float32),
            high=np.array([
                energy_config['max'],
                1.0,
                np.inf,
                np.inf
            ], dtype=np.float32),
            dtype=np.float32
        )
        self.logger.info(f"Defined observation space: low={self.observation_space.low}, high={self.observation_space.high}")

        # Metadata for Gymnasium (optional, but recommended)
        self.metadata = {"render_modes": [], "render_fps": 4}

        # Internal State
        self.init: bool = False
        self.rng = np.random.default_rng()
        self.avg_price: float = 0.0
        self.energy_lvl: float = energy_config['init']
        self.reward_type: int = 0
        self.count: int = 0        # Step counter
        self.terminated: bool = False
        self.truncated: bool = False

        # Extract other configurations if necessary
        self.pricing_eta: float = self.env_config.get('pricing', {}).get('eta', 0.5)
        self.time_steps_per_day_ratio: int = self.env_config.get('time', {}).get('time_steps_per_day_ratio', 1440)
        self.time_step_duration: float = self.env_config.get('time', {}).get('step_duration', 5)  # in minutes
        self.max_steps_per_episode: int = self.env_config.get('time', {}).get('max_steps_per_episode', 288)

        # Initialize the Reward Function
        self.logger.info(f"Setting up reward function: {reward_type}")
        self.reward: BaseReward = self._initialize_reward(reward_type)
        
        
        self.logger.info("EnergyNetEnv initialization complete.")
        
    def _initialize_reward(self, reward_type: str) -> BaseReward:
        """
        Initializes the reward function based on the specified type.

        Args:
            reward_type (str): Type of reward ('cost').

        Returns:
            BaseReward: An instance of a reward class.
        
        Raises:
            ValueError: If an unsupported reward_type is provided.
        """
        if reward_type == 'cost':
            return CostReward()
        
        else:
            self.logger.error(f"Unsupported reward type: {reward_type}")
            raise ValueError(f"Unsupported reward type: {reward_type}")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads a YAML configuration file.

        Args:
            config_path (str): Path to the YAML config file.

        Returns:
            Dict[str, Any]: Configuration parameters.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        if not os.path.exists(config_path):
            self.logger.error(f"Configuration file not found at {config_path}")
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, 'r') as file:
            config: Dict[str, Any] = yaml.safe_load(file)
            self.logger.debug(f"Loaded configuration from {config_path}: {config}")

        return config

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
        self.logger.info("Resetting environment.")

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.logger.debug(f"Random number generator seeded with: {seed}")
        else:
            self.rng = np.random.default_rng()
            self.logger.debug("Random number generator initialized without seed.")

        # Reset PCSUnit and ISO
        self.PCSUnit.reset()
        self.ISO.reset()
        self.logger.debug("PCSUnit and ISO have been reset.")

        # Reset internal state
        energy_config: Dict[str, Any] = self.pcs_unit_config['battery']['model_parameters']
        self.avg_price = 0.0
        self.energy_lvl = energy_config['init']
        self.reward_type = 0  # Default reward type

        # Handle options
        if options and 'reward' in options:
            if options.get('reward') == 1:
                self.reward_type = 1
                self.logger.debug("Reward type set to 1 based on options.")
            else:
                self.logger.debug(f"Reward type set to {self.reward_type} based on options.")
        else:
            self.logger.debug("No reward type option provided; using default.")

        # Reset step counter
        self.count = 0
        self.terminated = False
        self.truncated = False
        self.init = True

        # Initialize current time (fraction of day)
        current_time: float = (self.count * self.time_step_duration) / 1440  # 1440 minutes in a day
        self.logger.debug(f"Initial time set to {current_time} fraction of day.")

        # Update PCSUnit with current time and no action
        self.PCSUnit.update(time=current_time, battery_action=0.0)
        self.logger.debug("PCSUnit updated with initial time and no action.")

        # Fetch self-production and self-consumption
        self_production: float = self.PCSUnit.get_self_production()
        self_consumption: float = self.PCSUnit.get_self_consumption()
        self.logger.debug(f"Initial self-production: {self_production}, self-consumption: {self_consumption}")

        # Create initial observation
        observation: np.ndarray = np.array([
            self.energy_lvl,
            current_time,
            self_production,
            self_consumption
        ], dtype=np.float32)
        self.logger.debug(f"Initial observation: {observation}")

        info: Dict[str, float] = self._get_info()
        self.logger.debug(f"Initial info: {info}")

        return (observation, info)

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
        # Ensure the environment has been initialized
        assert self.init, "Environment must be reset before stepping."
        
        # Validate and process the action
        if isinstance(action, np.ndarray):
            if self.multi_action and action.shape != (3,):
                raise ValueError(f"Action numpy array must have shape (3,), but got shape {action.shape}.")
            elif not self.multi_action and action.shape != (1,):
                raise ValueError(f"Action numpy array must have shape (1,), but got shape {action.shape}.")

            
            # Check if the action is within the action space
            if not self.action_space.contains(action):
                raise ValueError(f"Action {action} is outside the action space {self.action_space}.")
            
            # Extract the scalar values from the array
            if self.multi_action:
                battery_action, consumption_action, production_action = action
                self.logger.debug(f"Actions extracted from numpy array: battery_action={battery_action}, consumption_action={consumption_action}, production_action={production_action}")
            else:
                battery_action = action.item()
                self.logger.debug(f"Action extracted from numpy array: battery_action={battery_action}")
                consumption_action = None
                production_action = None
    
        elif isinstance(action, float):
            if self.multi_action:
                raise TypeError("Expected a numpy array with shape (3,) for multi_action=True.")
            battery_action = action
            self.logger.debug(f"Action received as float: battery_action={battery_action}")
            consumption_action = None
            production_action = None
        else:
            raise TypeError(f"Invalid action type: {type(action)}. Action must be a float or a numpy array with shape (1,) or (3,).")          
            
        self.logger.debug(f"Stepping environment with actions: battery_action={battery_action}, consumption_action={consumption_action}, production_action={production_action}")

        # Increment step counter
        self.count += 1
        self.logger.debug(f"Step count incremented to {self.count}")

        # Calculate current time (fraction of day)
        current_time: float = (self.count * self.time_step_duration) / self.time_steps_per_day_ratio
        self.logger.debug(f"Current time set to {current_time} fraction of day.")

        # Update PCSUnit with current time and actions
        if self.multi_action:
            self.PCSUnit.update(time=current_time,
                                battery_action=battery_action,
                                consumption_action=consumption_action,
                                production_action=production_action)
        else:
            self.PCSUnit.update(time=current_time, battery_action=battery_action)
        self.logger.debug("PCSUnit updated with current actions.")

        # Fetch self-production and self-consumption
        self_production: float = self.PCSUnit.get_self_production()
        self_consumption: float = self.PCSUnit.get_self_consumption()
        self.logger.debug(f"Self-production: {self_production}, Self-consumption: {self_consumption}")

        # Determine how battery action affects net exchange
        if battery_action > 0:
            # Charging the battery uses energy, treated as additional consumption
            battery_charge = battery_action
            battery_discharge = 0.0
            # Net exchange = consumption (including battery charge) - production
            net_exchange = (self_consumption + battery_charge) - self_production
        elif battery_action < 0:
            # Discharging the battery provides additional supply
            battery_discharge = abs(battery_action)
            battery_charge = 0.0
            # Net exchange = consumption - (production + battery discharge)
            net_exchange = self_consumption - (self_production + battery_discharge)
        else:
            # No battery change
            battery_charge = 0.0
            battery_discharge = 0.0
            net_exchange = self_consumption - self_production

        self.logger.debug(f"Net exchange with grid: {net_exchange} (Positive=Buy, Negative=Sell)")

        # Prepare observation dictionary for ISO
        observation_dict: Dict[str, Any] = {
            'energy_level': self.energy_lvl,
            'current_price': None,
            'time': current_time,
            'self_production': self_production,
            'self_consumption': self_consumption
        }

        # Get pricing function from ISO (one price now)
        pricing_function = self.ISO.get_pricing_function(observation_dict)
        self.logger.debug("Retrieved pricing function from ISO.")


        price = pricing_function(1)
        self.logger.debug(f"Single price retrieved: {price} $/MWh")

        # Info dictionary
        info: Dict[str, Any] = {
            'net_exchange': net_exchange,
            'price': price,
            'pricing_function': pricing_function,
            # Add more info as needed for reward computation
        }
        self.logger.debug(f"Info for reward computation: {info}")

        # Compute reward using cost_reward logic
        # Reward is negative of cost: cost = net_exchange * price
        # If net_exchange > 0 (buy), cost >0 => reward negative
        # If net_exchange < 0 (sell), cost <0 => reward positive
        reward: float = self.reward.compute_reward(info)
        self.logger.debug(f"Calculated reward: {reward} based on net_exchange: {net_exchange}")

        # Update energy level from PCSUnit's battery state
        self.energy_lvl = self.PCSUnit.battery.get_state()
        self.logger.debug(f"Updated energy level: {self.energy_lvl} MWh")

        # Update running average price
        total_transactions = abs(net_exchange)  # total energy transacted with the grid
        if total_transactions > 0:
            avg_transaction_price: float = (price * net_exchange) / total_transactions
            # avg_transaction_price might represent cost per MWh (could refine logic if needed)
            self.logger.debug(f"Average transaction price: {avg_transaction_price}")
        else:
            avg_transaction_price = 0.0
            self.logger.debug("No transactions occurred; average transaction price set to 0.0")

        self.avg_price = (1.0 - self.pricing_eta) * self.avg_price + self.pricing_eta * avg_transaction_price
        self.logger.debug(f"Updated running average price: {self.avg_price}")

        # Determine if the episode is done
        done: bool = self.count >= self.max_steps_per_episode
        self.logger.debug(f"Episode done status: {done}")

        terminated: bool = done
        truncated: bool = False

        # Create next observation
        observation: np.ndarray = np.array([
            self.energy_lvl,
            current_time,
            self_production,
            self_consumption
        ], dtype=np.float32)
        self.logger.debug(f"Next observation: {observation}")

        return (observation, float(reward), terminated, truncated, info)

    def _get_info(self) -> Dict[str, float]:
        """
        Provides additional information about the environment's state.

        Returns:
            Dict[str, float]: Dictionary containing the running average price.
        """
        return {"running_avg": self.avg_price}


    def close(self):
        """
        Cleanup method. Closes loggers and releases resources.
        """
        self.logger.info("Closing environment.")

        # Close loggers if necessary
        # Example:
        logger_names = ['EnergyNetEnv', 'Battery', 'ProductionUnit', 'ConsumptionUnit', 'PCSUnit'] 
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        self.logger.info("Environment closed successfully.")