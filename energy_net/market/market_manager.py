from energy_net.model.state import State


class SimpleMarketManager():
    def __init__(self, name: str, market):
        self.mame = name
        self.market = market

    def step(self, cur_state:State):
        market_results = self.market.step(cur_state)
        return market_results

