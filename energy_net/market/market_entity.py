from abc import abstractmethod
from ..model.state import State
from ..model.action import EnergyAction
from ..model.reward import Reward
from ..defs import Bid
from energy_net.network_entity import NetworkEntity
from ..strategic_entity import StrategicEntity


class MarketEntity(StrategicEntity):
    def __init__(self, name, network_entity:NetworkEntity):
        self.name = name
        self.network_entity = network_entity

    @abstractmethod
    def get_bid(self, bid_type:str, state:State, args)-> Bid:
        pass

    def step(self, action: EnergyAction) -> [State,Reward]:
        return self.network_entity.step(action)

    def predict(self, state: State, action: EnergyAction) -> [State, Reward]:
        return self.network_entity.predict(state, action)


class MarketProducer(MarketEntity):

    def __init__(self, name: str, network_entity:NetworkEntity):
        super().__init__(name, network_entity=NetworkEntity)

class MarketConsumer(MarketEntity):

    def __init__(self, name: str, network_entity:NetworkEntity):
        super().__init__(name, network_entity=NetworkEntity)



