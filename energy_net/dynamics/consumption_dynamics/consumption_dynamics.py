from numpy.typing import ArrayLike
import numpy as np
from typing import Union

from ..energy_dynamcis import ConsumptionDynamics
from ...model.action import ConsumeAction
from ...model.state import ConsumptionState, State
from ...data.data import TimeSeriesData
from ...utils.utils import move_time_tick, convert_hour_to_int

class GeneralLoad(ConsumptionDynamics):
    def __init__(self, file_name: str, value_row_name: str, time_row_name: str) -> None:
        super().__init__()
        self.data = TimeSeriesData(file_name)
        self.load_data = self.data.get_column(value_row_name)
        self.time_data = self.data.get_column(time_row_name).apply(convert_hour_to_int)
        self.max_electric_power = self.load_data.max()
        self.current_day_start_idx = None

    def do(self, action: Union[np.ndarray, ConsumeAction], state: ConsumptionState = None, params=None) -> ConsumptionState:
        """Get load consumption based on the current hour."""
        num_samples_per_day = 48
        if state.current_time_step % num_samples_per_day == 0:
            # Randomly select a new day (48 samples per day)
            num_days = len(self.load_data) // num_samples_per_day
            random_day = np.random.randint(0, num_days)
            self.current_day_start_idx = random_day * num_samples_per_day
        
        
        
        current_hour = int(state.hour)
        next_hour = current_hour + 1
        idx = self.current_day_start_idx + state.current_time_step % num_samples_per_day
        # hour_mask = (self.time_data >= current_hour) & (self.time_data < next_hour)
        load= self.load_data[idx]
        
        
        new_state = state.copy()
        assert isinstance(load, float), 'Invalid load data'
        if load < 0:
            load = 0
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


