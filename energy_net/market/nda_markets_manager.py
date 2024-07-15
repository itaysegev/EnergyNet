from energy_net.market.market_entity import MarketEntity
from energy_net.market.nda_market import NDAMarket
from energy_net.model.state import State
from energy_net.defs import Bid
from energy_net.utils.utils import condition, get_predicted_state


class NDAMarketsManager:

    def __init__(self, market_entities:list[MarketEntity], horizons:list[float]=[24,48] ):
        self.market_entities = market_entities
        self.horizons = horizons
        self.nda_markets = {}
        for horizon in horizons:
            self.nda_markets[horizon] = NDAMarket( self.market_entities)


    def step(self, cur_state:State):
        market_results = {}
        for horizon in self.horizons:
            predicted_state = get_predicted_state(cur_state,horizon)
            [demand, bids, workloads, price] = self.nda_markets.do_market_clearing(predicted_state)
            market_results[horizon] = [demand, bids, workloads, price]

        return market_results

