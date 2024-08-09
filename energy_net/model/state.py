import copy
from ..config import MIN_PRODUCTION, NO_CONSUMPTION, DEFAULT_INIT_POWER, DEFAULT_EFFICIENCY, INITIAL_TIME, MAX_PRODUCTION, MAX_ELECTRIC_POWER
from datetime import datetime

import numpy as np

class State:
    def __init__(self, current_time_step: int = 0, hour: int = 0):
        self.hour = hour
        self.current_time_step = current_time_step

    def get_timedelta_state(self, delta_hours):
        # Create a deep copy of the state
        timedelta_state = self.copy()
        # Update the hour, ensuring it wraps around correctly
        timedelta_state.hour = (self.hour + delta_hours) % 24
        timedelta_state.current_time_step += delta_hours
        return timedelta_state

    def to_numpy(self):
        # Convert current_time_step and hour to a NumPy array
        return np.array([self.current_time_step, self.hour], dtype=np.float32)

    @classmethod
    def from_numpy(cls, array):
        current_time_step = int(array[0])
        hour = int(array[1])
        return cls(current_time_step, hour)

    def copy(self):
        return copy.deepcopy(self)
    
    def get_hour(self):
        return self.hour
    
class GridState(State):
    def __init__(self, current_time_step: int = 0, hour: int = 0, price: float = 0.0):
        super().__init__(current_time_step, hour)
        self.price = price
        
    def to_numpy(self):
        base_array = super().to_numpy()
        grid_array = np.array([self.price], dtype=np.float32)
        return np.concatenate((base_array, grid_array))
    
    def copy(self):
        return GridState(self.current_time_step, self.hour, self.price)
    
    @classmethod
    def from_numpy(cls, array):
        base_dict = super(GridState, cls).from_numpy(array)
        return {
            **base_dict,
            'price': array[2]
        }

class StorageState(State):
    def __init__(self, current_time_step: int = 0, hour: int = 0, state_of_charge: float = DEFAULT_INIT_POWER, 
                 charging_efficiency: float = DEFAULT_EFFICIENCY, discharging_efficiency: float = DEFAULT_EFFICIENCY, 
                 power_capacity: float = DEFAULT_INIT_POWER, energy_capacity: float = DEFAULT_INIT_POWER):
        super().__init__(current_time_step, hour)
        self.state_of_charge = state_of_charge
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency
        self.power_capacity = power_capacity
        self.energy_capacity = energy_capacity

    def to_numpy(self):
        base_array = super().to_numpy()
        storage_array = np.array([self.state_of_charge, self.charging_efficiency, self.discharging_efficiency, 
                                  self.power_capacity, self.energy_capacity], dtype=np.float32)
        return np.concatenate((base_array, storage_array))

    @classmethod
    def from_numpy(cls, array):
        base_dict = super(StorageState, cls).from_numpy(array)
        return {
            **base_dict,
            'state_of_charge': array[2],
            'charging_efficiency': array[3],
            'discharging_efficiency': array[4],
            'power_capacity': array[5],
            'energy_capacity': array[6]
        }

class ProductionState(State):
    def __init__(self, current_time_step: int = 0, hour: int = 0, max_production: float = MAX_PRODUCTION, 
                 production: float = MIN_PRODUCTION):
        super().__init__(current_time_step, hour)
        self.max_production = max_production
        self.production = production

    def to_numpy(self):
        base_array = super().to_numpy()
        production_array = np.array([self.max_production, self.production], dtype=np.float32)
        return np.concatenate((base_array, production_array))

    @classmethod
    def from_numpy(cls, array):
        base_dict = super(ProductionState, cls).from_numpy(array)
        return {
            **base_dict,
            'max_production': array[2],
            'production': array[3]
        }

class ConsumptionState(State):
    def __init__(self, current_time_step: int = 0, hour: int =0, max_electric_power: float = MAX_ELECTRIC_POWER, 
                 consumption: float = NO_CONSUMPTION):
        super().__init__(current_time_step, hour)
        self.max_electric_power = max_electric_power
        self.consumption = consumption

    def to_numpy(self):
        base_array = super().to_numpy()
        consumption_array = np.array([self.max_electric_power, self.consumption], dtype=np.float32)
        return np.concatenate((base_array, consumption_array))

    @classmethod
    def from_numpy(cls, array):
        base_dict = super(ConsumptionState, cls).from_numpy(array)
        return {
            **base_dict,
            'max_electric_power': array[2],
            'consumption': array[3]
        }

    def copy(self):
        return ConsumptionState(self.current_time_step, self.hour, self.max_electric_power, self.consumption)

class PcsunitState(State):
    def __init__(self, states: dict):
        # Extract states from the dictionary based on their types
        consumption_state = next((state for state in states.values() if isinstance(state, ConsumptionState)), None)
        storage_state = next((state for state in states.values() if isinstance(state, StorageState)), None)
        production_state = next((state for state in states.values() if isinstance(state, ProductionState)), None)
        grid_state = next((state for state in states.values() if isinstance(state, GridState)), None)
        # Ensure all states are assigned and have the same current time_step and hour
        if not consumption_state or not storage_state or not production_state or not grid_state:
            raise ValueError("All four states (ConsumptionState, StorageState, ProductionState, GridState) must be provided.")

        assert consumption_state.current_time_step == storage_state.current_time_step == production_state.current_time_step == grid_state.current_time_step, "All states must have the same current time_step."
        assert consumption_state.hour == storage_state.hour == production_state.hour == grid_state.hour, "All states must have the same hour."

        super().__init__(current_time_step=consumption_state.current_time_step, hour=consumption_state.hour)
        self.consumption_state = consumption_state
        self.storage_state = storage_state
        self.production_state = production_state
        self.grid_state = grid_state

    def to_numpy(self):
        base_array = super().to_numpy()
        consumption_array = self.consumption_state.to_numpy()[2:]  # Exclude the current_time_step and hour
        storage_array = self.storage_state.to_numpy()[2:]  # Exclude the current_time_step and hour
        production_array = self.production_state.to_numpy()[2:]  # Exclude the current_time_step and hour
        grid_array = self.grid_state.to_numpy()[2:]  # Exclude the current_time_step and hour
        return np.concatenate((base_array, consumption_array, storage_array, production_array, grid_array), axis=0)

    @classmethod
    def from_numpy(cls, array):
        current_time_step = int(array[0])
        hour = int(array[1])
        consumption_state_dict = ConsumptionState.from_numpy(np.concatenate(([array[0], array[1]], array[2:5])))
        storage_state_dict = StorageState.from_numpy(np.concatenate(([array[0], array[1]], array[5:11])))
        production_state_dict = ProductionState.from_numpy(np.concatenate(([array[0], array[1]], array[11:13])))
        grid_state_dict = GridState.from_numpy(np.concatenate(([array[0], array[1]], array[13:14])))
        
        consumption_state = ConsumptionState(**consumption_state_dict)
        storage_state = StorageState(**storage_state_dict)
        production_state = ProductionState(**production_state_dict)
        grid_state = GridState(**grid_state_dict)
        return cls({'consumption_state': consumption_state, 'storage_state': storage_state, 'production_state': production_state, 'grid_state': grid_state})

    def copy(self):
        return PcsunitState({
            'consumption_state': self.consumption_state.copy(),
            'storage_state': self.storage_state.copy(),
            'production_state': self.production_state.copy(),
            'grid_state': self.grid_state.copy()
        })
        
    def get_price(self):
        return self.grid_state.price
    
    def get_production(self):
        return self.production_state.production
    
    def get_consumption(self):
        return self.consumption_state.consumption
    
    def get_soc(self):
        return self.storage_state.state_of_charge
    

    
    
