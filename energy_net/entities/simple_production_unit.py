from energy_net.entities.pcsunit import PCSUnit


class SimpleProductionUnit:
    def __init__(self, name: str, max_production: float, efficiency: float, production_dynamics, init_state):
        self.name = name
        self.max_production = max_production
        self.efficiency = efficiency
        self.production_dynamics = production_dynamics
        self.state = init_state

    def produce(self, amount):
        pass

    def step(self, cur_state):
        pass

    def get_state(self):
        return self.state

