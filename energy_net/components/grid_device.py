import numpy as np
from ..defs import Bounds
from .device import Device
from ..model.state import GridState
from ..model.action import ConsumeAction
from ..config import MIN_PRICE ,MAX_PRICE, INITIAL_TIME, MAX_TIME, MAX_HOUR, INITIAL_HOUR
from .params import DeviceParams

class GridDevice(Device):
    def __init__(self, params: DeviceParams):
        init_state = GridState(price=MIN_PRICE)
        super().__init__(params, init_state=init_state)


    # @property
    # def current_state(self) -> ConsumptionState:
    #     return ConsumptionState(max_electric_power=self.max_electric_power, efficiency=self.efficiency, consumption=self.consumption)
    
    # def update_state(self, state: ConsumptionState):
    #     self.max_electric_power = state.max_electric_power
    #     self.efficiency = state.efficiency
    #     self.consumption = state.consumption
    #     super().update_state(state)

  
    def reset(self) -> GridState:
        super().reset()
        
        
    def get_action_space(self):
        return Bounds(low=MIN_PRICE, high=MAX_PRICE, shape=(1,), dtype=np.float32)


    def get_observation_space(self):
        # Define the lower and upper bounds for each dimension of the observation space
        low = np.array([INITIAL_TIME, INITIAL_HOUR,  MIN_PRICE])  
        high = np.array([MAX_TIME, MAX_HOUR, MAX_PRICE]) 
        return Bounds(low=low, high=high, shape=(len(low),), dtype=np.float32)
        
    