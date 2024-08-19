from ...config import DEFAULT_PRODUCTION
from ..energy_dynamcis import  ProductionDynamics
from ...model.action import ProduceAction
from ...model.state import ProductionState
from ...data.data import TimeSeriesData
from ...utils.utils import move_time_tick, convert_hour_to_int
import numpy as np
from typing import Union
import pandas as pd

class PVDynamics(ProductionDynamics):
    def __init__(self, file_name: str, value_row_name: str, time_row_name: str) -> None:
        super().__init__()
        self.data = TimeSeriesData(file_name)
        self.solar_data = self.data.get_column(value_row_name)
        self.time_data = self.data.get_column(time_row_name).apply(convert_hour_to_int)
        self.max_production = self.solar_data.max()
        self.init_production = self.solar_data[0]
        self.current_day_start_idx = None

    def do(self, action: Union[np.ndarray, ProduceAction], state: ProductionState = None, params=None) -> ProductionState:
        """Get solar generation output based on the current hour."""

        num_samples_per_day = 48
        if state.current_time_step % num_samples_per_day == 0:
            # Randomly select a new day (48 samples per day)
            num_days = len(self.solar_data) // num_samples_per_day
            random_day = np.random.randint(0, num_days)
            self.current_day_start_idx = random_day * num_samples_per_day

        idx = self.current_day_start_idx + state.current_time_step % num_samples_per_day
        solar_data = self.solar_data[idx]
        state.max_production = self.max_production
        
        new_state = state.copy()
        assert isinstance(solar_data, float), 'Invalid solar data'
        assert solar_data >= 0, 'Invalid solar data'
        assert solar_data <= state.max_production, 'Invalid solar data'

        new_state.production = solar_data
        new_state.current_time_step, new_state.hour = move_time_tick(state.current_time_step, state.hour)
        return new_state
    
    def predict(self, action, params, state):
        pass

    def get_current_production_capability(self):
        pass

    def predict_production_capability(self, state):
        pass
    

