from typing import Any, Union, Mapping
import numpy as np


from ..model.action import EnergyAction
from energy_net.model.state import PcsunitState
from .network_entity import  CompositeNetworkEntity, NetworkEntity
from energy_net.devices.storage_devices.local_storage import Battery
from energy_net.defs import Bounds

from ..utils.utils import AggFunc

class PCSUnit(CompositeNetworkEntity):
    """ A network entity that contains a list of sub-entities. The sub-entities are the devices and the pcsunit itself is the composite entity.
    The PCSUnit entity is responsible for managing the sub-entities and aggregating the reward.
    """

    def __init__(self, name: str, sub_entities: dict[str, NetworkEntity] = None, agg_func: AggFunc = None):
        super().__init__(name, sub_entities, agg_func)

    def step(self, actions: Union[np.ndarray, EnergyAction], **kwargs) -> None:
        if isinstance(actions, np.ndarray):
            for entity in self.sub_entities.values():
                entity.step(actions)
        else:
            raise NotImplementedError
        
        
    def predict(self, actions: Union[np.ndarray, dict[str, EnergyAction]]):

        predicted_states = {}
        if type(actions) is np.ndarray:
            # we convert the entity dict to a list and match action to entities by index
            sub_entities = list(self.sub_entities.values())
            for entity_index, action in enumerate(actions):
                if hasattr(type(sub_entities[entity_index]), 'predict'):
                    predicted_states[sub_entities[entity_index].name] = sub_entities[entity_index].predict(
                    np.array([action]))

        else:
            for entity_name, action in actions.items():
                if hasattr(type(self.sub_entities[entity_name]), 'predict'):
                    predicted_states[entity_name] = self.sub_entities[entity_name].predict(action)

        if self.agg_func:
            agg_value = self.agg_func(predicted_states)
            return agg_value
        else:
            return predicted_states
        
    def get_observation_space(self) -> dict[str, Bounds]:
        obs_space = {}
        first_entity = True
        for name, entity in self.sub_entities.items():
            obs_space[name] = entity.get_observation_space()
            if first_entity:
                first_entity = False
            elif isinstance(entity, Battery):
                # Remove the time dimension from the observation space due to duplicates
                obs_space[name].remove_first_dim()
            else:
                obs_space[name].remove_first_dim()
                obs_space[name].remove_first_dim()
            
        return obs_space
        
    def get_action_space(self) -> dict[str, Bounds]:
        action_space = {}
        for name, entity in self.sub_entities.items():
            if isinstance(entity, Battery):
                action_space[name] = entity.get_action_space()
            
        return action_space
    
    def get_state(self, numpy_arr = False) -> dict[str, PcsunitState]:
        states = {}
        for entity in self.sub_entities.values():
            states[entity.name] = entity.get_state()
        
        state = PcsunitState(states)
        
        if numpy_arr:
            return state.to_numpy()
            
        return state







