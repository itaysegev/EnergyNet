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

class PCSUnit(CompositeNetworkEntity):
    """ A network entity that contains a list of sub-entities. The sub-entities are the devices and the pcsunit itself is the composite entity.
    The PCSUnit entity is responsible for managing the sub-entities and aggregating the reward.
    """
    def __init__(self, name: str, storage_device: Battery, consumer_device: ConsumerDevice, private_producer: PrivateProducer):
        self.name = name
        self.storage_device = storage_device
        self.consumer_device = consumer_device
        self.private_producer = private_producer
        super().__init__()

    def step(self, storage_action: StorageAction, consumption_action: ConsumeAction, production_action: ProduceAction):
        #   ------------------------------------------ storage action ------------------------------------------
        self.storage_device.step(storage_action)
        self.consumer_device.step(consumption_action)
        self.private_producer.step(production_action)

    def get_current_state(self) -> State:
        storage_state = self.storage_device.get_state()
        consumption_state = self.consumer_device.get_state()
        production_state = self.private_producer.get_state()
        return State(storage_state, consumption_state, production_state)


    def get_observation_space(self) -> Bounds:

        return NotImplemented


    def get_action_space(self) -> Bounds:
        storage_devices_action_sapces = [v.get_action_space() for v in self.get_storage_devices().values()]
        
        # Combine the Bounds objects into a single Bound object
        combined_low = np.array([bound['low'] for bound in storage_devices_action_sapces])
        combined_high = np.array([bound['high'] for bound in storage_devices_action_sapces])
        return Bounds(low=combined_low, high=combined_high, shape=(len(combined_low),),  dtype=np.float32)


    def reset(self) -> State:
        # return self.apply_func_to_sub_entities(lambda entity: entity.reset())
        self._state = self._init_state
        self._state['pred_consumption'] = self.predict_next_consumption()
        for entity in self.sub_entities.values():
            entity.reset()
        return self._state


    def get_storage_devices(self):
        return self.apply_func_to_sub_entities(lambda entity: entity, condition=lambda x: isinstance(x, Battery))
    
    def validate_action(self, actions: dict[str, EnergyAction]):
        for entity_name, action in actions.items():
            if len(action) > 1 or 'charge' not in action.keys():
                raise ValueError(f"Invalid action key {action.keys()} for entity {entity_name}")
            else:
                return True


    def apply_func_to_sub_entities(self, func, condition=lambda x: True):
        results = {}
        for name, entity in self.sub_entities.items():
            if condition(entity):
                results[name] = func(entity)
        return results

    def get_next_consumption(self) -> float:
        return sum([self.sub_entities[name].predict_next_consumption() for name in self.consumption_keys])





