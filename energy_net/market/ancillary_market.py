from energy_net.market.market_entity import MarketEntity


class AncillaryMarket():
    def __init__(self, market_entities:list[MarketEntity]):
        self.market_entities = market_entities
