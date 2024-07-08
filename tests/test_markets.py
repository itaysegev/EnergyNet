
import warnings

from energy_net import NDAMarketsManager
from energy_net.dynamics.production_dynamics.productionUnit_dynamics import ProductionUnitDynamics
from energy_net.energy_market import EnergyMarket
from energy_net.entities.consumers_agg import ConsumersAgg
from energy_net.entities.production_unit import ProductionUnit
from energy_net.model.action import EnergyAction, ProduceAction, StorageAction, ConsumeAction
from energy_net.dynamics.consumption_dynamics.consumption_dynamics import ConsumptionDynamics
from energy_net.devices.params import ConsumptionParams, ProductionParams
from energy_net.config import DEFAULT_LIFETIME_CONSTANT
from energy_net.model.state import State


def test_markets():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        init_state = State()
        # initialize consumers aggregator
        consumption_params = ConsumersAgg(name='agg_consumption', energy_dynamics=ConsumptionDynamics(),init_state=init_state)

        # initialize producers
        producers_arr=[]
        production_unit = ProductionUnit(name='test_producer1', max_production=100, efficiency=0.9, energy_dynamics=ProductionUnitDynamics())
        producers_arr.append(production_unit)
        production_unit = ProductionUnit(name='test_producer2', max_production=100, efficiency=0.9, energy_dynamics=ProductionUnitDynamics())
        producers_arr.append(production_unit)
        production_unit = ProductionUnit(name='test_producer3', max_production=100, efficiency=0.9, energy_dynamics=ProductionUnitDynamics())
        producers_arr.append(production_unit)

        # nda_markets_manager
        nda_markets_manager = NDAMarketsManager()

        # initilaze energy market
        energy_market = EnergyMarket("test_market", nda_markets_manager=nda_markets_manager)

        # run markets

if __name__ == '__main__':
    test_markets()

