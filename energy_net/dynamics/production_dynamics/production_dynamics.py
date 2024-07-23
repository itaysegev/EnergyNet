from ...config import DEFAULT_PRODUCTION
from ..energy_dynamcis import  ProductionDynamics
from ...model.action import ProduceAction
from ...model.state import ProductionState
from ...data.data import TimeSeriesData
from ...utils.utils import move_time_tick
import numpy as np
from typing import Union
import pandas as pd

class PVDynamics(ProductionDynamics):
    def __init__(self, file_name: str, value_row_name: str, time_row_name: str) -> None:
        super().__init__()
        self.data = TimeSeriesData(file_name)
        self.solar_data = self.data.get_column(value_row_name)
        self.time_data = self.data.get_column(time_row_name)
        self.max_production = self.solar_data.max()

    def do(self, action: Union[np.ndarray, ProduceAction], state: ProductionState = None, params=None) -> ProductionState:
        """Get solar generation output based on the current hour."""
        current_hour = int(state.hour)
        next_hour = current_hour + 1
        hour_mask = (self.time_data >= current_hour) & (self.time_data < next_hour)
        solar_data_within_hour = self.solar_data[hour_mask]
        state.max_production = self.max_production
        
        assert len(solar_data_within_hour) > 0, 'No solar data within the current hour'
        solar_data = np.random.choice(solar_data_within_hour)  # Sample from the data within the current hour
        
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
    

