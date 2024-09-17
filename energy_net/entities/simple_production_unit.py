import copy
from abc import abstractmethod
from collections import OrderedDict
from typing import Union
import numpy as np

from ..dynamics.energy_dynamcis import EnergyDynamics
from ..utils.utils import AggFunc
from ..model.action import EnergyAction
from ..model.state import State
from ..model.reward import Reward
from energy_net.defs import Bounds
from energy_net.entities.network_entity import NetworkEntity, ElementaryNetworkEntity


class SimpleProductionUnit(ElementaryNetworkEntity):
    def __init__(self, name, energy_dynamics: EnergyDynamics , init_state:State, max_production, efficiency):
        super().__init__(name, energy_dynamics, init_state)
        # if the state is none - this is a stateless entity

    def produce(self, amount):
        pass

    def step(self, cur_state):
        pass

    def get_state(self):
        return self.state

