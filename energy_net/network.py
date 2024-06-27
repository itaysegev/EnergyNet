from .entities.network_entity import NetworkEntity
from typing import Union
import numpy as np
from energy_net.model.action import EnergyAction
from energy_net.network_manager import NetworkManager
from energy_net.defs import Bounds



class Network():
    def __init__(self, name: str, network_entities: list[NetworkEntity], market_network: Union[NetworkEntity, None] = None,
                physical_network: Union[NetworkEntity, None] = None, network_manager: Union[NetworkManager, None] = None ) -> None:
        self.network_entities = network_entities
        self.market_network = market_network
        self.physical_network = physical_network
        self.network_manager = network_manager
        self.name = name

    def step(self, action: Union[np.ndarray, EnergyAction]):
        """
        Advances the simulation by one time step.
        This method should update the state of each network entity.
        """
        for entity in self.network_entities:
            entity.step()

    def reset(self):
        """
        Resets the state of the network and all its entities to their initial state.
        This is typically used at the beginning of a new episode.
        """
        for entity in self.network_entities:
            entity.reset()

    def add_entity(self, entity: NetworkEntity):
        """
        Adds a new entity to the network.
        """
        self.network_entities.append(entity)


    def get_state(self) -> np.ndarray:
        """
        Returns the current state of the network.
        """
        state = []
        for entity in self.network_entities:
            state.append(entity.get_state())

        return np.array(state)   

    def get_observation_space(self) -> Bounds:
        """
        Returns the observation space of the network.
        """
        obs_space = []
        for entity in self.network_entities:
            obs_space.append(entity.get_observation_space())
        
        return obs_space
        
        
    
    def get_action_space(self) -> Bounds:
        """
        Returns the action space of the network.
        """
        action_space = []
        for entity in self.network_entities:
            action_space.append(entity.get_state())
        
        return action_space
    
    # def get_observation_names(self):
    #     """
    #     Returns the names of the observations in the observation space.
    #     """
    #     state = []
    #     for entity in self.network_entities:
    #         state.append(entity.get_state())


    
    # def get_action_names(self):
    #     """
    #     Returns the names of the actions in the action space.
    #     """
    #     return None
    
    # def get_episode_time_steps(self):
    #     """
    #     Returns the number of time steps in an episode.
    #     """
    #     return None