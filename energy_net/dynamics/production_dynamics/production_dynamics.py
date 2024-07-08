from ...config import DEFAULT_PRODUCTION
from ..energy_dynamcis import  ProductionDynamics
from ...model.action import EnergyAction
from ...model.state import ProducerState
from ...data.data import TimeSeriesData
from numpy.typing import ArrayLike
import pandas as pd

class PVDynamics(ProductionDynamics):
    def __init__(self) -> None:
        super().__init__()

    # TODO: Need to check which of the funciton to use for PV.
    #  PV is usually data-driven, although we can compute the raw power production as function of irradiation, location, time and etc.
    def do(self, action: EnergyAction, state:ProducerState=None , params= None) -> ProducerState:

        """Get solar generation output.
        """
        value = action['produce']
        if value is not None:
           new_state = state.copy()
           new_state['production'] = min(value, state['max_produce'])
           return new_state
        else:
            return self.get_current_production(state,params)

    def do_data_driven(self, time_step, action: EnergyAction = None, state: ProducerState = None, params = None) -> ProducerState:

        """Get solar generation output.
        """
        data = TimeSeriesData('CAISO_net-load_2021.xlsx')
        solar_data = data.get_column('Solar')

        return solar_data[time_step]



    def get_current_production(self, state, params):
        return DEFAULT_PRODUCTION
    
    def predict(self, action, params, state):
        pass

    def get_current_production_capability(self):
        pass

    def predict_production_capability(self, state):
        pass
    

