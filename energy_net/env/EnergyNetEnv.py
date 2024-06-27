from functools import lru_cache
from pathlib import Path
from typing import Any, List, Union

import numpy as np
from gymnasium.spaces import Box, Dict
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from ..network import Network
from ..network_agent import Agent
from ..config import DEFAULT_TIME_STEP
from ..defs import Bounds
from ..entities.pcsunit import DefaultPCSUnitRewardFunction
from ..env.base import Environment, EpisodeTracker
from ..model.action import EnergyAction
from ..model.reward import RewardFunction
from ..network import Network
from ..utils.env_utils import bounds_to_gym_box


class EnergyNetEnv(ParallelEnv, Environment):

    ##################
    # Pettingzoo API #
    ##################

    metadata = {"name": "energy_net_env_v0"}

    def __init__(self,
        network_lst: List[Network],
        simulation_start_time_step: int = None,
        simulation_end_time_step: int = None,
        episode_time_steps: int = None, 
        seconds_per_time_step: float = None,
        initial_seed: int = None,
        **kwargs: Any):

        self.episode_tracker = EpisodeTracker(simulation_start_time_step, simulation_end_time_step)
        super().__init__(seconds_per_time_step=seconds_per_time_step, random_seed=initial_seed, episode_tracker=self.episode_tracker)

        self.network_lst = network_lst 
        self.num_entities = len(self.network_lst)


        self.__state = None

        # set random seed if specified
        self.__np_random = None
        self.seed(initial_seed)

        # pettingzoo required attributes
        self.entities = {entity.name: entity for entity in self.network_lst}
        
        self.agents = []
        self.agents_name_to_network = {}
        
       
        # # reset episode tracker to start after initializing episode time steps during reset
        # self.episode_tracker.reset_episode_index()

        self.__observation_space = self.get_observation_space()
        self.__action_space = self.get_action_space()

        

    def reset(self, seed=None, return_info=True, options=None):
        
        self.reset_time_step()

        # set seed if given
        if seed is not None:
            self.seed(seed)

        assert len(self.agents) > 0, "No agents have been set. Please set agents before calling reset."

        for entity in self.entities.values():
            entity.reset()


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
            curr_state = self.agents_name_to_network[agent_name].get_state()
            # NEW TIME TICK

            self.self.agents_name_to_network.step(actions)
            #s'
            next_state = self.agents_name_to_network[agent_name].get_state()    
            
            #TODO: r
            # rewards[agent_name] = self.reward_function.calculate(curr_state, actions, next_state, time_steps=self.time_step)

        # get new observations according to the current state
        obs = self.__observe_all()

        self.__action_space = self.get_action_space()

        infos = self.get_info()

        #TODO: 
        # Check if the simulation has reached the end
        truncs = {a: False for a in self.agents}
        if self.terminated():
            truncs = {a: True for a in self.agents}
            terminations = {a: True for a in self.agents}
            self.agents = []

        self.next_time_step()

        return obs, rewards, terminations, truncs, infos

 
    @lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self.__observation_space[agent]

   
    def action_space(self, agent: str):
        return self.__action_space[agent]
    
    

    ######################
    # End Pettingzoo API #
    ######################


    #######################
    # Extra API Functions #
    #######################

    def set_agents(self, agent: Agent, network_idx: int):
        network = self.network_lst[network_idx]
        agent.set_network(network)
        self.agents_name_to_network[agent.name] = network
        self.agents.append(agent)
    
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
        return {agent: np.array(list(self.entities[agent].get_state().values()),dtype=np.float32) for agent in self.agents}

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


    def terminated(self):
        return self.time_step == self.simulation_end_time_step - self.simulation_start_time_step - 1
    

    @property
    def time_steps(self) -> int:
        """Number of time steps in current episode split."""
        return self.episode_tracker.episode_time_steps
    

    def get_info(self):
        return {agent: {} for agent in self.agents}
        






