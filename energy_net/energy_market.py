from energy_net.model.state import State


class EnergyMarket():
    def __init__(self, name: str, nda_markets_manager):
        self.mame = name
        self.nda_markets_manager = nda_markets_manager

    def step(self, cur_state:State):
        nda_market_reults = self.nda_markets_manager.step(cur_state)
        return nda_market_reults