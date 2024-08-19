from functools import lru_cache
from pathlib import Path
from typing import Any, List, Union

import numpy as np
from gymnasium.spaces import Box, Dict
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from ..network import Network
from ..defs import Bounds
from ..env.base import Environment, EpisodeTracker
from ..model.action import EnergyAction
from ..model.reward import RewardFunction
from ..network import Network

from ..utils.env_utils import bounds_to_gym_box, assign_indexes


class EnergyNetEnv(Environment, ParallelEnv):

    ##################
    # Pettingzoo API #
    ##################

    metadata = {"name": "energy_net_env_v0"}

    def __init__(self,
        network : Network,
        simulation_start_time_step: int = None, # Time step to start simulation
        simulation_end_time_step: int = None, 
        episode_time_steps: int = None, # Number of time steps in an episode
        seconds_per_time_step: float = None, # Number of seconds in 1 `time_step` and must be set to >= 1.
        initial_seed: int = None, #  Pseudorandom number generator seed for repeatable results.
        **kwargs: Any):

        self.episode_tracker = EpisodeTracker(simulation_start_time_step, simulation_end_time_step)
        super().__init__(seconds_per_time_step=seconds_per_time_step, random_seed=initial_seed, episode_tracker=self.episode_tracker)

        self.episode_time_steps = episode_time_steps


        # set random seed if specified
        self.__np_random = None
        self.seed(initial_seed)
        
        
        # initialize env configurations with argument or default value
        self.network = network
        #self._single_value_config 


        # pettingzoo required attributes
        self.agents = []
        self.possible_agents = list(self.network.strategic_entities.keys())
        
        
        # default rewards table set
        
        
        # set all custom reward tables for each agent
        
        
        # set final reward table for each agent as the default reward table updated by the custom rewards.
        
        
        
        
        
        
        
        # initialize observation and action spaces
        
        self.__observation_space = self.get_observation_space()
        self.__action_space = self.get_action_space()
        

    
        # state and env objects
        self.__state = None

    def reset(self, seed=None, return_info=True, options=None):
        
        super().reset()

        # update time steps for time series
        self.episode_tracker.next_episode(
            self.episode_time_steps,
            False,
            False,
            self.random_seed,
        )


        # set seed if given
        if seed is not None:
            self.seed(seed)

        
        self.info = {'action_space': self.__action_space, 'observation_space': self.__observation_space}
        # reset network
        self.network.reset()
        
        # reset agents
        self.agents = list(self.network.strategic_entities.keys())
        
        

        
        self.__state = self.network.get_state()

        self.__action_space = self.get_action_space()

        # get all observations
        observations = self.__observe_all()
        
        
        if not return_info:
            return observations
        else:
            return observations, self.get_info()

    def seed(self, seed=None):
        self.__np_random, seed = seeding.np_random(seed)
        
        
    def state(self):
        return self.__state.copy()
    
    
    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Box:
        return self.__observation_space[agent]

    # @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Box:
        return self.__action_space[agent]

    def step(self, joint_action: dict[str, Union[np.ndarray, EnergyAction]]):
        
        
        # step in environment with sampled action
        new_state, rewards, truncs, infos = self.network.step(joint_action)
        
        
        # log desired actions and performed transitions
        for agent, agent_info in infos.items():
            agent_info['desired_action'] = joint_action[agent]
            
            # agent_info['performed_transition'] = true_actions[agent]
        
        # set new state in environment
        self.__state = new_state

        # get new observations according to the current state
        obs = self.__observe_all()
        
        # remove done agents from live agents list
        for agent_name, done in truncs.items():
            if done:
                self.agents.remove(agent_name)

        self.__action_space = self.get_action_space()

        # infos = self.get_info()

        # #TODO: 
        # # Check if the simulation has reached the end
        terms = {a: False for a in self.agents}
        if self.terminated():
            terms = {a: True for a in self.agents}
            self.agents = []

        self.next_time_step()

        return obs, rewards, terms, truncs, infos


    ######################
    # End Pettingzoo API #
    ######################


    #######################
    # Extra API Functions #
    #######################

    # def set_agents(self, agent: NetworkAgent, network_idx: int) -> None:
    #     network = self.network_lst[network_idx]
    #     agent.set_network(network)
    #     self.agents_name_to_network[agent.name] = network
    #     self.agents.append(agent)
    
    
    
    
    def agent_iter(self):
        """
        Returns an iterator over all agents.
        """
        return iter(self.agents)
    
    
    def set_state(self, state):
        """
        Sets the current environment state

        Args:
            state: the state to set in the environment
        """
        self.__state = state
        
    def get_state(self):
        """
        Returns the current environment state
        """
        return self.__state


    def observe_all(self):
        """
        gets all agents observations for the given state.
        This is an API exposure of an inner method.

        Returns:
            a dictionary for all agents observations.
        """
        return self.__observe_all()
    
    
    
    def __observe_all(self):
        return {agent: self.network.stratigic_to_network_entity[agent].get_state(numpy_arr=True) for agent in self.agents}

    # def convert_space(self, space):
    #     if isinstance(space, dict):
    #         return Dict(space)
    #     elif isinstance(space, Bounds):
    #         return Box(low=space.low, high=space.high, shape=(1,), dtype=space.dtype)
    #     else:
    #         raise TypeError("observation space not supported")

    def get_observation_space(self) -> dict[str, Box]:
        return {name: bounds_to_gym_box(bound) for name, bound in self.network.get_observation_space().items()}
    

    def get_action_space(self) -> dict[str, Box]:
        return {name: bounds_to_gym_box(bound) for name, bound in self.network.get_action_space().items()}


    def terminated(self) -> bool:
        return self.time_step == self.episode_tracker.simulation_end_time_step - self.episode_tracker.simulation_start_time_step - 1
    

    def truncated(self) -> bool:
        """Check if episode truncates due to a time limit or a reason that is not defined as part of the task MDP."""

        return False

    @property
    def time_steps(self) -> int:
        """Number of time steps in current episode split."""
        return self.episode_tracker.episode_time_steps
    

    @property
    def episode(self) -> int:
        """Current episode index."""

        return self.episode_tracker.episode
    
    def get_info(self) -> dict:
        return {agent: {} for agent in self.agents}
    
    
    
    
    
    

    





