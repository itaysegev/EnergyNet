from energy_net.network_entity import ElementaryNetworkEntity
from energy_net.dynamics.energy_dynamcis import ProductionDynamics
from energy_net.defs import Bounds
from energy_net.model.state import State

class ProductionUnit(ElementaryNetworkEntity):
    """ A network entity that contains a list of sub-entities. The sub-entities are the devices and the pcsunit itself is the composite entity.
    The PCSUnit entity is responsible for managing the sub-entities and aggregating the reward.
    """
    def __init__(self, name, production_dynamics: ProductionDynamics , init_state:State, max_production:float, efficiency:float):
        super().__init__(name, energy_dynamics=production_dynamics, init_state=init_state)
        self.max_production = max_production
        self.efficiency = efficiency


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





