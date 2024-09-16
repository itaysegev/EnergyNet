from ...config import DEFAULT_PRODUCTION
from ..energy_dynamcis import  ProductionDynamics
from ...model.action import ProduceAction
from ...model.state import ProductionState
from ...data.data import TimeSeriesData
from ...utils.utils import move_time_tick, convert_hour_to_int
import numpy as np
from typing import Union
import pandas as pd

class SimpleProductionDynamics(ProductionDynamics):
    def __init__(self):
        super().__init__()

    def do(self, action: Union[np.ndarray, ProduceAction], state: ProductionState = None, params=None) -> ProductionState:
        pass
