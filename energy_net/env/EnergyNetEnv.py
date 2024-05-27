from functools import lru_cache
from pathlib import Path
from typing import Any, List, Union

import numpy as np
from gymnasium.spaces import Box, Dict
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from ..config import DEFAULT_TIME_STEP
from ..defs import Bounds
from ..entities.pcsunit import DefaultPCSUnitRewardFunction
from ..env.base import Environment, EpisodeTracker
from ..model.action import EnergyAction
from ..model.reward import RewardFunction
from ..network_entity import NetworkEntity
from ..utils.env_utils import bounds_to_gym_box


class EnergyNetEnv(ParallelEnv, Environment):

    ##################
    # Pettingzoo API #
    ##################

    metadata = {"name": "energy_net_env_v0"}

    def __init__(self,
        network_entities: List[NetworkEntity],
        root_directory: Union[str, Path] = None, 
        simulation_start_time_step: int = None,
        simulation_end_time_step: int = None,
        episode_time_steps: int = None, 
        seconds_per_time_step: float = None,
        initial_seed: int = None,
        reward_function: RewardFunction = None,
        **kwargs: Any):

        # set the root directory
        self.root_directory = root_directory
        self.episode_tracker = EpisodeTracker(simulation_start_time_step, simulation_end_time_step)
        super().__init__(seconds_per_time_step=seconds_per_time_step, random_seed=initial_seed, episode_tracker=self.episode_tracker)


        self.network_entities = network_entities 
        self.timestep = None
        self.episode_time_steps = episode_time_steps
        self.simulation_start_time_step = simulation_start_time_step
        self.simulation_end_time_step = simulation_end_time_step
        self.time_step_num = simulation_end_time_step - simulation_start_time_step if simulation_end_time_step is not None and simulation_start_time_step is not None else DEFAULT_TIME_STEP
        self.num_entities = len(self.network_entities)
        

        # set random seed if specified
        self.__np_random = None
        self.seed(initial_seed)

        # pettingzoo required attributes
        self.entities = {entity.name: entity for _, entity in enumerate(self.network_entities)}
        self.possible_agents = list(self.entities.keys())
        self.agents = []
        
        # set reward function
        self.reward_function = reward_function(env_metadata=self.get_metadata) if reward_function is not None else DefaultPCSUnitRewardFunction(env_metadata=self.get_metadata)
       
        # reset environment and initializes episode time steps
        self.reset()

        # reset episode tracker to start after initializing episode time steps during reset
        self.episode_tracker.reset_episode_index()

        self.__observation_space = self.get_observation_space()
        self.__action_space = self.get_action_space()

        # state and env objects
        self.__state = None
        # self.__rewards = None
        # self.__episode_rewards = []
        
        
        



    def reset(self, seed=None, return_info=True, options=None):
        
        self.reset_time_step()
        # set seed if given
        if seed is not None:
            self.seed(seed)


        # reset agents
        self.agents = self.possible_agents.copy()
        

        for entity in self.entities.values():
            entity.reset()

        # reset reward function (does nothing by default)
        self.reward_function.reset()
        self.__action_space = self.get_action_space()
        # get all observations
        observations = self.__observe_all()
        
        
        if not return_info:
            return observations
        else:
            return observations, self.get_info()

    def seed(self, seed=None):
        self.__np_random, seed = seeding.np_random(seed)


    def step(self, joint_action: dict[str, Union[np.ndarray, EnergyAction]]):

        # init array of rewards and termination flag per agent
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}

        # Perform the actions
        for agent_name, actions in joint_action.items():
            #s
            curr_state = self.entities[agent_name].get_current_state()
            # NEW TIME TICK

            self.entities[agent_name].step(actions)
            #s'
            next_state = self.entities[agent_name].get_current_state()    
            #r
            rewards[agent_name] = self.reward_function.calculate(curr_state, actions, next_state, time_steps=self.time_step)

        # get new observations according to the current state
        obs = self.__observe_all()
        self.__action_space = self.get_action_space()
        infos = self.get_info()
        # Check if the simulation has reached the end
        truncs = {a: False for a in self.agents}
        if self.terminated():
            truncs = {a: True for a in self.agents}
            terminations = {a: True for a in self.agents}
            self.agents = []

        self.next_time_step()

        return obs, rewards, terminations, truncs, infos

    '''

    @abstractmethod
    def get_action_space(self) -> spaces:
        """
        Get the action space of the network entity.

        Returns:
        spaces: The action space.
        """
        pass

    @abstractmethod
    def get_observation_space(self) -> spaces:
        """
        Get the observation space of the network entity.

        Returns:
        spaces: The observation space.
        """
        pass

    '''
    @lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self.__observation_space[agent]

   
    def action_space(self, agent: str):
        return self.__action_space[agent]
    
    @property
    def possible_agents(self):
        return self.__possible_agents
    
    @possible_agents.setter
    def possible_agents(self, possible_agents: List[str]):
        self.__possible_agents = possible_agents
    

    ######################
    # End Pettingzoo API #
    ######################


    #######################
    # Extra API Functions #
    #######################
    
    def agent_iter(self):
        """
        Returns an iterator over all agents.
        """
        return iter(self.agents)


    def observe_all(self):
        """
        gets all agents observations for the given state.
        This is an API exposure of an inner method.

        Returns:
            a dictionary for all agents observations.
        """
        return self.__observe_all()
    
    def __observe_all(self):
        return {agent: np.array(list(self.entities[agent].get_current_state().values()),dtype=np.float32) for agent in self.agents}

    def convert_space(self, space):
        if isinstance(space, dict):
            return Dict(space)
        elif isinstance(space, Bounds):
            return Box(low=space.low, high=space.high, shape=(1,), dtype=space.dtype)
        else:
            raise TypeError("observation space not supported")

    def get_observation_space(self):
        return {name: bounds_to_gym_box(entity.get_observation_space()) for name, entity in self.entities.items()}
    

    def get_action_space(self):
        return {name: bounds_to_gym_box(entity.get_action_space()) for name, entity in self.entities.items()}



    @property
    def episode_rewards(self):
        return self.__episode_rewards
    
    @episode_rewards.setter
    def episode_rewards(self, episode_rewards):
        self.__episode_rewards = episode_rewards
    
    @property
    def rewards(self):
        """reward time series for each agents"""
        return self.__rewards
    
    @rewards.setter
    def rewards(self, rewards):
        self.__rewards = rewards


    def terminated(self):
        return self.time_step == self.simulation_end_time_step - self.simulation_start_time_step - 1
    

    @property
    def time_steps(self) -> int:
        """Number of time steps in current episode split."""
        return self.episode_tracker.episode_time_steps
    


    def get_info(self):
        return {agent: {} for agent in self.agents}
        






