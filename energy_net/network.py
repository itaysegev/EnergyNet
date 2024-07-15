from energy_net import market_manager
from energy_net.network_entity import NetworkEntity
from typing import Union
import numpy as np
from energy_net.model.action import EnergyAction

from energy_net.defs import Bounds



class Network():
    def __init__(self): #name: str, strategic_entities: dict[str, NetworkEntity], market_network: Union[list[NetworkEntity], None] = None,physical_network: Union[NetworkEntity, None] = None, market_manager: MarketManager=None, None] = None) -> None:
        #self.strategic_entities = strategic_entities
        #self.market_manager = market_manager
        #self.physical_network = physical_network
        #self.network_manager = network_manager
        #self.name = name
        pass

    def step(self, joint_actions: dict[str, EnergyAction]):
        """
        Advances the simulation by one time step.
        This method should update the cur_state of each network entity.
        """
        for agent_name, action  in joint_actions.items():
            self.strategic_entities[agent_name].step(action)
            
    def reset(self):
        """
        Resets the cur_state of the network and all its entities to their initial cur_state.
        This is typically used at the beginning of a new episode.
        """
        for entity in self.strategic_entities.values():
            entity.reset()


    def get_state(self) -> np.ndarray:
        """
        Returns the current cur_state of the network.
        """
        state = []
        for entity in self.network_entities:
            state.append(entity.get_state())

        return np.array(state)   

    def get_observation_space(self) -> dict[str, Bounds]:
        """
        Returns the observation space of the network.
        """
        return {agent_name: entity.get_observation_space() for agent_name, entity in self.strategic_entities.items()}
        
        
    
    def get_action_space(self) -> dict[str, Bounds]:
        """
        Returns the action space of the network.
        """ 
        return {agent_name: entity.get_action_space() for agent_name, entity in self.strategic_entities.items()}
        
    
  