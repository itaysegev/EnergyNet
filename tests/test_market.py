
import warnings

from energy_net.dynamics.production_dynamics.productionUnit_dynamics import ProductionUnitDynamics
from energy_net.market.market_entity import MarketEntity
from energy_net.market.nda_market import NDAMarket
from energy_net.market_manager import MarketManager
from energy_net.entities.consumers_agg import ConsumersAgg
from energy_net.entities.production_unit import ProductionUnit
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
        # initialize consumers aggregator
        consumption = ConsumersAgg(name='agg_consumption', consumption_dynamics=ConsumptionDynamics(),init_state=init_state)

        # initialize producers
        market_producers_arr=[]
        production_unit = ProductionUnit(name='_producer1', max_production=100, efficiency=0.9, production_dynamics=ProductionUnitDynamics(), init_state=init_state)
        market_producer = MarketEntity(name='market'+production_unit.name, network_entity=production_unit)
        market_producers_arr.append(market_producer)
        production_unit = ProductionUnit(name='_producer2', max_production=100, efficiency=0.9, production_dynamics=ProductionUnitDynamics(), init_state=init_state)
        market_producer = MarketEntity(name='market'+production_unit.name, network_entity=production_unit)
        market_producers_arr.append(market_producer)
        production_unit = ProductionUnit(name='_producer3', max_production=100, efficiency=0.9, production_dynamics=ProductionUnitDynamics(), init_state=init_state)
        market_producer = MarketEntity(name='market'+production_unit.name, network_entity=production_unit)
        market_producers_arr.append(market_producer)

        # nda_market
        nda_market =  NDAMarket(consumption_entities=[consumption], production_entities =market_producers_arr, horizons = [24], intervals = [0.5])

        # market_manager
        market_manager = MarketManager(name="market_manager", markets=[nda_market])

        # run market
        cur_state = init_state
        market_manager.step(cur_state=cur_state)

if __name__ == '__main__':
    test_market()

