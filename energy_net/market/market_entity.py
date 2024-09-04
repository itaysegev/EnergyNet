class MarketEntity:
    def get_state(self):
        return self.network_entity.get_state()

    def take_action(self, action):
        pass



class ProductionMarketEntity(MarketEntity):
    def __init__(self, name: str, network_entity):
        self.name = name
        self.network_entity = network_entity

    def step(self, cur_state):
        pass

    def bid(self, description, state, demand):
        # placeholder
        bid = 100
        return bid

