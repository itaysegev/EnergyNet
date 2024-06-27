from numpy.typing import ArrayLike

from ..dynamics.energy_dynamcis import ConsumptionDynamics
from ..model.action import EnergyAction
from ..model.state import ConsumerState, State


class ElectricHeaterDynamics(ConsumptionDynamics):
    def __init__(self) -> None:
        super().__init__()
        
    def do(self, action: EnergyAction, state:ConsumerState) -> ConsumerState:
        """Get electric heater consumption.
        
        Parameters
        ----------
        action : ArrayLike
            Action to be performed. Must be a numpy array.
        state : HeaterState
            Current state of the electric heater.
        return : float
            Electric heater consumption in [kW].
        """
        value = action[0]
        if value is not None:
            new_state = state.copy()
            new_state.consumption = min(value, state.max_electric_power)
            return new_state	
        else:
            raise ValueError('Invalid action')

    
    def predict(self, action, params, state):
        pass

    def get_current_consumption_capability(self):
        pass

    def predict_consumption_capability(self, state):
        pass


