import warnings

from energy_net.dynamics.production_dynamics.simple_production import SimpleProductionDynamics
from energy_net.market.market_entity import ProductionMarketEntity
from energy_net.market.nda_market import NDAMarket
from energy_net.market_manager import SimpleMarketManager
from energy_net.entities.consumers_agg import ConsumersAgg
from energy_net.entities.simple_production_unit import SimpleProductionUnit
from energy_net.model.action import EnergyAction, ProduceAction, StorageAction, ConsumeAction
from energy_net.dynamics.consumption_dynamics.consumption_dynamics import ConsumptionDynamics
from energy_net.devices.params import ConsumptionParams, ProductionParams
from energy_net.config import DEFAULT_LIFETIME_CONSTANT
from energy_net.model.state import State
from datetime import datetime


def test_market():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        init_state = State(datetime.now())
        horizon = 24
        interval = 0.5
        # initialize consumers aggregator
        consumption = ConsumersAgg(name='agg_consumption', consumption_dynamics=ConsumptionDynamics(),init_state=init_state)

        # initialize producers
        market_producers_arr=[]
        production_unit = SimpleProductionUnit(name='_producer1', max_production=100, efficiency=0.9, production_dynamics=SimpleProductionDynamics(), init_state=init_state)
        market_producer = ProductionMarketEntity(name='market'+production_unit.name, network_entity=production_unit)
        market_producers_arr.append(market_producer)
        production_unit = SimpleProductionUnit(name='_producer2', max_production=100, efficiency=0.9, production_dynamics=SimpleProductionDynamics(), init_state=init_state)
        market_producer = ProductionMarketEntity(name='market'+production_unit.name, network_entity=production_unit)
        market_producers_arr.append(market_producer)
        production_unit = SimpleProductionUnit(name='_producer3', max_production=100, efficiency=0.9, production_dynamics=SimpleProductionDynamics(), init_state=init_state)
        market_producer = ProductionMarketEntity(name='market'+production_unit.name, network_entity=production_unit)
        market_producers_arr.append(market_producer)

        # nda_market
        nda_market =  NDAMarket(consumption_entities=[consumption], production_entities=market_producers_arr, horizons=[horizon], intervals=[interval])

        # market_manager
        market_manager = SimpleMarketManager(name="market_manager", market=nda_market)

        # run market
        cur_state = init_state
        for i in range(int(horizon/interval)):
            market_manager.step(cur_state=cur_state)


if __name__ == '__main__':
    test_market()

