from energy_net.entities.network_entity import ElementaryNetworkEntity
from energy_net.dynamics.energy_dynamcis import ConsumptionDynamics
from energy_net.defs import Bounds
from energy_net.model.state import State


class ConsumersAgg(ElementaryNetworkEntity):

    def __init__(self, name, consumption_dynamics: ConsumptionDynamics , init_state:State):
        super().__init__(name, energy_dynamics=consumption_dynamics, init_state=init_state)

    def get_observation_space(self) -> dict[str, Bounds]:
        pass

    def get_action_space(self) -> dict[str, Bounds]:
        pass