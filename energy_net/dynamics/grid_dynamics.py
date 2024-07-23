from numpy.typing import ArrayLike
import numpy as np
from typing import Union

from .energy_dynamcis import EnergyDynamics
from ..model.state import GridState
from ..utils.utils import move_time_tick, hourly_pricing

class GridDynamics(EnergyDynamics):
    def __init__(self) -> None:
        super().__init__()

        
    def do(self, action: np.ndarray, state:GridState) -> GridState:
       
        # load = self.load_data[state.current_time]

        new_state = state.copy()
        new_state.price = hourly_pricing(state.hour)
        new_state.current_time_step, new_state.hour = move_time_tick(state.current_time_step, state.hour)
        return new_state
        

    def predict(self, action, params, state):
        pass



