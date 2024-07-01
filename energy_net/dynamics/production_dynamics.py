from ..config import DEFAULT_PRODUCTION
from ..dynamics.energy_dynamcis import  ProductionDynamics
from ..model.action import EnergyAction
from ..model.state import ProducerState
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

    # TODO: Not sure if I can pass time_step as a parameter or maybe its inside the state?
    '''
     def do_data_driven(self, action: EnergyAction, state: ProducerState = None, time_step, params=None) -> ProducerState:

        """Get solar generation output.
        """
        df = pd.read_csv('solar_production_real_data.csv')

        production_data = df.Solar.to_numpy()

        production_data = [int(x.replace(',', '')) if ',' in x else x for x in production_data]

        state['production'] = production_data[time_step]

        return state
    '''


    def get_current_production(self, state, params):
        return DEFAULT_PRODUCTION
    
    def predict(self, action, params, state):
        pass

    def get_current_production_capability(self):
        pass

    def predict_production_capability(self, state):
        pass
    

