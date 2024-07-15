from energy_net.model.state import State


class MarketManager():
    def __init__(self, name: str, markets):
        self.mame = name
        self.markets = markets

    def step(self, cur_state:State):
        markets_results = []
        for market in self.markets:
            market_reults = market.step(cur_state)
            markets_results.append(market_reults)
        return markets_results