from energy_net.market.market_entity import MarketEntity
from energy_net.model.state import State
from energy_net.defs import Bid
from energy_net.utils.utils import condition, get_predicted_state


class MarketManager:
    def __init__(self, market_entities:list[MarketEntity]):
        self.market_entities = market_entities



    def run(self, initial_state:State, stop_criteria:condition, horizons:list[float]=[24,48]):
        cur_state = initial_state
        while not stop_criteria(cur_state):
            for horizon in horizons:
                predicted_state = get_predicted_state(cur_state,horizon)
                [demand, bids, workloads, price] = self.do_market_clearing(predicted_state)
                #check solution validity

                #send solution



