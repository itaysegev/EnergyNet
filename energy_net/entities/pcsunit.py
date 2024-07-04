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
        storage_state = self.storage_device.get_observation_space()
        consumption_state = self.consumer_device.get_observation_space()
        production_state = self.private_producer.get_observation_space()
        return NotImplemented


    def get_action_space(self) -> Bounds:
        storage_state = self.storage_device.get_action_space()
        consumption_state = self.consumer_device.get_action_space()
        production_state = self.private_producer.get_action_space()
        return NotImplemented


    def reset(self) -> State:
        self.storage_device.reset()
        self.consumer_device.reset()
        self.private_producer.reset()





