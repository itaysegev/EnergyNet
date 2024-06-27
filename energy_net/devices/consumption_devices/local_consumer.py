
import numpy as np
from ...defs import Bounds
from ..device import Device
from ...model.state import ConsumerState
from ...model.action import ConsumeAction
from ...config import MAX_ELECTRIC_POWER, MIN_POWER, MIN_EFFICIENCY, MAX_EFFICIENCY, NO_CONSUMPTION
from ...entities.params import ConsumptionParams

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
        self.max_electric_power = consumptionParams["max_electric_power"] if "max_electric_power" in consumptionParams is None else MAX_ELECTRIC_POWER
        self.efficiency = consumptionParams["efficiency"] if "efficiency" in consumptionParams else MAX_EFFICIENCY
        self.init_max_electric_power = self.max_electric_power
        self.consumption = NO_CONSUMPTION
        self.action_type = ConsumeAction
        init_state = ConsumerState(max_electric_power=self.max_electric_power, efficiency=self.efficiency, consumption=self.consumption)
        super().__init__(consumptionParams, init_state=init_state)


    @property
    def max_electric_power(self):
        return self._max_electric_power
    
    @max_electric_power.setter
    def max_electric_power(self, max_electric_power: float):
        assert max_electric_power >= MIN_POWER, 'max_electric_power must be >= MIN_POWER.'
        self._max_electric_power = max_electric_power

    @property
    def current_state(self) -> ConsumerState:
        return ConsumerState(max_electric_power=self.max_electric_power, efficiency=self.efficiency)
    
    @property
    def efficiency(self) -> float:
        return self._efficiency
    
    @efficiency.setter
    def efficiency(self, efficiency: float):
        assert efficiency >= MIN_EFFICIENCY, 'efficiency must be >= MIN_EFFICIENCY.'
        self._efficiency = efficiency
    
    def get_current_state(self) -> ConsumerState:
        return self.current_state
    

    def update_state(self, state: ConsumerState):
        self.max_electric_power = state.max_electric_power
        self.efficiency = state.efficiency
        self.consumption = state.consumption
        super().update_state(state)

  
    def reset(self) -> ConsumerState:
        super().reset()
        self.max_electric_power = self.init_max_electric_power
        self.consumption = NO_CONSUMPTION
        


    def get_action_space(self):
        return Bounds(low=MIN_POWER, high=self.max_electric_power, shape=(1,), dtype=np.float32)


    def get_observation_space(self):
        # Define the lower and upper bounds for each dimension of the observation space
        low = np.array([NO_CONSUMPTION, MIN_POWER, MIN_EFFICIENCY])  # Example lower bounds
        high = np.array([self.max_electric_power, MAX_ELECTRIC_POWER, MAX_EFFICIENCY])  # Example upper bounds
        return Bounds(low=low, high=high, shape=(len(low),), dtype=np.float32)
        
    