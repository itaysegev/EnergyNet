from energy_net.dynamics.energy_dynamcis import EnergyDynamics
from energy_net.model.state import State
from energy_net.grid_entity import ElementaryGridEntity


class ProductionUnit(ElementaryGridEntity):
    def __init__(self, name, energy_dynamics: EnergyDynamics , init_state:State, max_production, efficiency):
        super().__init__(name, energy_dynamics, init_state)
        # if the state is none - this is a stateless entity

    def produce(self, amount):
        pass

    def step(self, cur_state):
        pass

    def get_state(self):
        return self.state

