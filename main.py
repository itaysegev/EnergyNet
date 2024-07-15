from energy_net.defs import Bid, State
from energy_net.market_entity import MarketProducer, MarketConsumer
from energy_net.network_entity import ElementaryNetworkEntity
from energy_net.dynamics.energy_dynamcis import ProductionDynamics, ConsumptionDynamics
from energy_net.market.nda_markets_manager import NDAMarketsManager
from energy_net.utils import agg_func_sum


class DummyStationDynamics(ProductionDynamics):
    def __init__(self, production_capacity):
        self.production_capacity = production_capacity

    def do(self, action, params):
        if action == "produce":
            return min(params['amount'], self.production_capacity)
        else:
            raise NotImplemented('stations can only produce')

    def predict(self, action, params, state):
        if action == "produce":
            return min(params['amount'], self.production_capacity)
        elif action == "consumption":
            return 0
        else:
            raise NotImplemented('stations can only produce')


class DummyProducer(ElementaryNetworkEntity):
    def __init__(self, name, energy_dynamics: ProductionDynamics):
        super().__init__(name, energy_dynamics=energy_dynamics)

class DummyMarketProducer(MarketProducer):
    def __init__(self, name, energy_dynamics: ProductionDynamics, fixed_price):
        super().__init__(name, network_entity=DummyProducer(name, energy_dynamics))
        self.fixed_price = fixed_price

    def get_bid(self, bid_type, state, args) -> Bid:
        if bid_type=='production':
            bid = [self.energy_dynamics.production_capacity, self.fixed_price]
            return bid
        else:
            return None

    def predict(self, action, params, state):
        return self.network_entity.predict(action, params, state)


class DummyConsumptionDynamics(ConsumptionDynamics):
    def __init__(self, fixed_consumption):
        self.consumption = fixed_consumption

    def do(self, action, params):
        if action == 'consume':
            return self.consumption
        else:
            raise NotImplemented('consumers can only consume')

    def predict(self, action, params, state):
        if action == "consumption":
            return self.consumption
        else:
            raise NotImplemented('consumers can only consume')

class DummyMarketConsumer(MarketConsumer):
    def __init__(self, name, energy_dynamics: ConsumptionDynamics):
        super().__init__(name, energy_dynamics)

    def get_bid(self, bid_type, state, args):
        return None

def test_elementary_network_market():
    station1 = DummyMarketProducer(name= 'station1', energy_dynamics=DummyStationDynamics(production_capacity=200), fixed_price=10)
    station2 = DummyMarketProducer('station2', DummyStationDynamics(production_capacity=300), fixed_price=2)
    station3 = DummyMarketProducer('station3', DummyStationDynamics(production_capacity=400), fixed_price=4)
    station4 = DummyMarketProducer('station4', DummyStationDynamics(production_capacity=500), fixed_price=100)
    station5 = DummyMarketProducer('station5', DummyStationDynamics(production_capacity=600), fixed_price=42)

    consumer1 = DummyConsumer('consumer1', DummyConsumptionDynamics(fixed_consumption=200))
    consumer2 = DummyConsumer('consumer2', DummyConsumptionDynamics(fixed_consumption=400))
    consumer3 = DummyConsumer('consumer3', DummyConsumptionDynamics(fixed_consumption=600))
    consumer4 = DummyConsumer('consumer4', DummyConsumptionDynamics(fixed_consumption=8))


    mgr = NDAMarketsManager([station1, station2, station3, station4, station5, consumer1, consumer2, consumer3, consumer4])
    demand = mgr.collect_demand(None)
    bids = mgr.collect_production_bids(None, demand)
    workloads, price = mgr.market_clearing_merit_order(demand, bids)

    print(f'demand was: {demand}')
    print(f'bids were: {bids}')
    print(f'workloads are: {workloads}')
    print(f'final price is: {price}')




def test_composite_network_market():
    station1 = DummyMarketProducer('station1', DummyStationDynamics(production_capacity=200), fixed_price=10)
    station2 = DummyMarketProducer('station2', DummyStationDynamics(production_capacity=300), fixed_price=2)
    station3 = DummyMarketProducer('station3', DummyStationDynamics(production_capacity=400), fixed_price=4)
    station4 = DummyMarketProducer('station4', DummyStationDynamics(production_capacity=500), fixed_price=100)
    station5 = DummyMarketProducer('station5', DummyStationDynamics(production_capacity=600), fixed_price=42)

    consumer1 = DummyMarketConsumer('consumer1', DummyConsumptionDynamics(fixed_consumption=20))
    consumer2 = DummyMarketConsumer('consumer2', DummyConsumptionDynamics(fixed_consumption=400))
    consumer3 = DummyMarketConsumer('consumer3', DummyConsumptionDynamics(fixed_consumption=60))
    consumer4 = DummyMarketConsumer('consumer4', DummyConsumptionDynamics(fixed_consumption=800))

    # this is the key difference from the elementary implementation
    consumer_aggregator = DummyConsumerAggregator('agg1', [], agg_func_sum)
    mgr = NDAMarketsManager([station1, station2, station3, station4, station5, consumer_aggregator])
    demand = mgr.collect_demand(None)
    bids = mgr.collect_production_bids(None, demand)
    workloads, price = mgr.market_clearing('merit_order',demand, bids)

    print(f'demand was: {demand}')
    print(f'bids were: {bids}')
    print(f'workloads are: {workloads}')
    print(f'final price is: {price}')

def test_market_run():
    station1 = DummyMarketProducer('station1', DummyStationDynamics(production_capacity=200), fixed_price=10)
    station2 = DummyMarketProducer('station2', DummyStationDynamics(production_capacity=300), fixed_price=2)
    station3 = DummyMarketProducer('station3', DummyStationDynamics(production_capacity=400), fixed_price=4)
    station4 = DummyMarketProducer('station4', DummyStationDynamics(production_capacity=500), fixed_price=100)
    station5 = DummyMarketProducer('station5', DummyStationDynamics(production_capacity=600), fixed_price=42)

    consumer1 = DummyMarketConsumer('consumer1', DummyConsumptionDynamics(fixed_consumption=20))
    consumer2 = DummyMarketConsumer('consumer2', DummyConsumptionDynamics(fixed_consumption=400))
    consumer3 = DummyMarketConsumer('consumer3', DummyConsumptionDynamics(fixed_consumption=60))
    consumer4 = DummyMarketConsumer('consumer4', DummyConsumptionDynamics(fixed_consumption=800))

    # this is the key difference from the elementary implementation
    mgr = NDAMarketsManager([station1, station2, station3, station4, station5, consumer1, consumer2, consumer3, consumer4])
    state = State({'time':0})
    print(mgr.run(state))

if __name__ == "__main__":
    #test_elementary_network_market()
    #test_composite_network_market()
    test_market_run()

#############################

#implementing a neighborhood of 5 buildings
# building 1: driven by a data file that determines what happens at each step (like CityLearn)
# building 2: is modeled as a single unit (no sub-entities) that has a solar pannel and a batery
# building 3: a building modeled as 4 appartments (sub-entities), all of which consume energy, and 2 have solar pannels and batteries
# building 4: a building that models 4 appartments, 2 are modeled as a single unit, 1 has several consumer devices, and 1 is data driven.
# building 5: a house with many devices, each device can only consume, but the house has a single battery.


#b1 = ...


#b2 = ...


#b3 = ...


#b4 = ...


#b5 = ...



#my_neighborhood = NetworkEntity(sub_entities=[b1, b2, b3, b4, b5])