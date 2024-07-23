
import numpy as np
from ...defs import Bounds
from ..device import Device
from ...model.state import ConsumptionState
from ...model.action import ConsumeAction
from ...config import MAX_ELECTRIC_POWER, MIN_POWER, INITIAL_HOUR, MAX_HOUR, MIN_EFFICIENCY, MAX_EFFICIENCY, NO_CONSUMPTION, INITIAL_TIME, MAX_TIME
from ..params import ConsumptionParams

class ConsumerDevice(Device):
    """Base consumer class.
    Parameters
    ----------
    max_electric_power : float, default: None
        Maximum amount of electric power that the electric heater can consume from the power grid.


    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, consumptionParams:ConsumptionParams):
        self.max_electric_power = consumptionParams["max_electric_power"] 
        self.init_max_electric_power = self.max_electric_power
        self.consumption = NO_CONSUMPTION
        self.action_type = ConsumeAction
        init_state = ConsumptionState(max_electric_power=self.max_electric_power, consumption=self.consumption)
        super().__init__(consumptionParams, init_state=init_state)


    @property
    def max_electric_power(self):
        return self._max_electric_power
    
    @max_electric_power.setter
    def max_electric_power(self, max_electric_power: float):
        assert max_electric_power >= MIN_POWER, 'max_electric_power must be >= MIN_POWER.'
        self._max_electric_power = max_electric_power

    @property
    def current_state(self) -> ConsumptionState:
        return ConsumptionState(max_electric_power=self.max_electric_power, consumption=self.consumption)
    

    def reset(self) -> ConsumptionState:
        super().reset()
        self.max_electric_power = self.init_max_electric_power
        self.consumption = NO_CONSUMPTION
        


    def get_action_space(self):
        return Bounds(low=MIN_POWER, high=self.max_electric_power, shape=(1,), dtype=np.float32)


    def get_observation_space(self):
        # Define the lower and upper bounds for each dimension of the observation space
        low = np.array([INITIAL_TIME, INITIAL_HOUR, MIN_POWER, NO_CONSUMPTION,])  # Example lower bounds
        high = np.array([MAX_TIME, MAX_HOUR, MAX_ELECTRIC_POWER, self.max_electric_power])  # Example upper bounds
        return Bounds(low=low, high=high, shape=(len(low),), dtype=np.float32)
        
    