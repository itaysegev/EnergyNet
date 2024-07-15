from numpy.typing import ArrayLike

from ..energy_dynamcis import ConsumptionDynamics
from ...model.action import EnergyAction
from ...model.state import ConsumptionState, State


class ElectricHeaterDynamics(ConsumptionDynamics):
    def __init__(self) -> None:
        super().__init__()
        
    def do(self, action: EnergyAction, state:ConsumptionState) -> ConsumptionState:
        """Get electric heater consumption.
        
        Parameters
        ----------
        action : ArrayLike
            Action to be performed. Must be a numpy array.
        state : HeaterState
            Current cur_state of the electric heater.
        return : float
            Electric heater consumption in [kW].
        """
        value = action["consume"] 
        if value is not None:
            new_state = state.copy()
            new_state['consumption'] = min(value, state['max_electric_power'])
            return new_state	
        else:
            raise ValueError('Invalid action')

    
    def predict(self, action, params, state):
        pass

    def get_current_consumption_capability(self):
        pass

    def predict_consumption_capability(self, state):
        pass


