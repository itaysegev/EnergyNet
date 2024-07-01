from ..config import DEFAULT_PRODUCTION
from ..dynamics.energy_dynamcis import  ProductionDynamics
from ..model.action import EnergyAction
from ..model.state import ProducerState
from numpy.typing import ArrayLike

class PVDynamics(ProductionDynamics):
    def __init__(self) -> None:
        super().__init__()

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
        

    def get_current_production(self, state, params):
        return DEFAULT_PRODUCTION
    
    def predict(self, action, params, state):
        pass

    def get_current_production_capability(self):
        pass

    def predict_production_capability(self, state):
        pass
    

