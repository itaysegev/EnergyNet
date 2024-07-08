from energy_net.network_entity import ElementaryNetworkEntity
from energy_net.dynamics.energy_dynamcis import ConsumptionDynamics
from ..model.state import State


class ConsumersAgg(ElementaryNetworkEntity):

    def __init__(self, name, consumption_dynamics: ConsumptionDynamics , init_state:State):
        super().__init__(name, energy_dynamics=consumption_dynamics, init_state=init_state)

    def get_observation_space(self) -> dict[str, Bounds]:
        return obs_space

    def get_action_space(self) -> dict[str, Bounds]:
        return action_space