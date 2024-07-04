from typing import Any, Union, Mapping
import numpy as np

from ..config import INITIAL_TIME, NO_CONSUMPTION, MAX_CONSUMPTION, NO_CHARGE, MAX_CAPACITY, PRED_CONST_DUMMY, MIN_POWER
from ..defs import Bounds
from ..model.action import EnergyAction, StorageAction, TradeAction, ConsumeAction, ProduceAction

from ..model.state import State
from .network_entity import  CompositeNetworkEntity
from ..devices.storage_devices.local_storage import Battery
from ..devices.consumption_devices.local_consumer import ConsumerDevice

from ..devices.params import StorageParams, ProductionParams, ConsumptionParams
from ..devices.production_devices.local_producer import PrivateProducer

from ..utils.utils import AggFunc

class PCSUnit(CompositeNetworkEntity):
    """ A network entity that contains a list of sub-entities. The sub-entities are the devices and the pcsunit itself is the composite entity.
    The PCSUnit entity is responsible for managing the sub-entities and aggregating the reward.
    """

    def __init__(self, name: str, sub_entities: dict[str, Battery, ConsumerDevice, PrivateProducer] = None, agg_func: AggFunc = None):
        super().__init__(name)
        self.sub_entities = sub_entities
        self.agg_func = agg_func

    def step(self, actions: dict[str, Union[np.ndarray, EnergyAction]], **kwargs) -> None:
        for entity_name, action in actions.items():
            if type(action) is np.ndarray:
                action = self.sub_entities[entity_name].action_type.from_numpy(action)

            self.sub_entities[entity_name].step(action, **kwargs)

    def predict(self, actions: Union[np.ndarray, dict[str, EnergyAction]]):

        predicted_states = {}
        if type(actions) is np.ndarray:
            # we convert the entity dict to a list and match action to entities by index
            sub_entities = list(self.sub_entities.values())
            for entity_index, action in enumerate(actions):
                predicted_states[sub_entities[entity_index].name] = sub_entities[entity_index].predict(
                    np.array([action]))

        else:
            for entity_name, action in actions.items():
                predicted_states[entity_name] = self.sub_entities[entity_name].predict(action)

        if self.agg_func:
            agg_value = self.agg_func(predicted_states)
            return agg_value
        else:
            return predicted_states

    def get_state(self) -> dict[str, State]:
        state = {}
        for entity in self.sub_entities.values():
            state[entity.name] = entity.get_state()

        if self.agg_func:
            state = self.agg_func(state)

        return state

    def reset(self) -> None:
        for entity in self.sub_entities.values():
            entity.reset()

    def get_observation_space(self) -> dict[str, Bounds]:
        obs_space = {}
        for name, entity in self.sub_entities.items():
            obs_space[name] = entity.get_observation_space()

        return obs_space

    def get_action_space(self) -> dict[str, Bounds]:
        action_space = {}
        for name, entity in self.sub_entities.items():
            action_space[name] = entity.get_action_space()

        return action_space





