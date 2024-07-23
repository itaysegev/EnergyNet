from numpy.typing import ArrayLike
import numpy as np
from typing import Union

from ..energy_dynamcis import ConsumptionDynamics
from ...model.action import ConsumeAction
from ...model.state import ConsumptionState, State
from ...data.data import TimeSeriesData
from ...utils.utils import move_time_tick

class GeneralLoad(ConsumptionDynamics):
    def __init__(self, file_name: str, value_row_name: str, time_row_name: str) -> None:
        super().__init__()
        self.data = TimeSeriesData(file_name)
        self.load_data = self.data.get_column(value_row_name)
        self.time_data = self.data.get_column(time_row_name)
        self.max_electric_power = self.load_data.max()

    def do(self, action: Union[np.ndarray, ConsumeAction], state: ConsumptionState = None, params=None) -> ConsumptionState:
        """Get load consumption based on the current hour."""
        current_hour = int(state.hour)
        next_hour = current_hour + 1
        hour_mask = (self.time_data >= current_hour) & (self.time_data < next_hour)
        load_data_within_hour = self.load_data[hour_mask]
        
        if len(load_data_within_hour) == 0:
            load = 0.0  # Default value if no data is found
        else:
            load = np.random.choice(load_data_within_hour)  # Sample from the data within the current hour
        
        new_state = state.copy()
        assert isinstance(load, float), 'Invalid load data'
        assert load >= 0, 'Invalid load data'
        assert load <= state.max_electric_power, 'Invalid load data'

        new_state.consumption = load
        new_state.current_time_step, new_state.hour = move_time_tick(state.current_time_step, state.hour)
        return new_state
        


    def predict(self, action, params, state):
        pass

    def get_current_consumption_capability(self):
        pass

    def predict_consumption_capability(self, state):
        pass


