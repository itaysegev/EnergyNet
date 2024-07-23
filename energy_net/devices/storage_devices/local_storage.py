'''This code is based on https://github.com/intelligent-environments-lab/CityLearn/blob/master/citylearn/energy_model.py'''

import numpy as np

from ..params import StorageParams
from ...defs import Bounds
from ...model.action import StorageAction
from ..device import Device
from ...config import MIN_CHARGE, MIN_EFFICIENCY, MAX_EFFICIENCY, MIN_CAPACITY, MAX_CAPACITY, INITIAL_TIME, MAX_TIME
from ...model.state import StorageState
from ..params import DeviceParams, StorageParams
from ...config import DEFAULT_EFFICIENCY, NO_CHARGE, MAX_CAPACITY, MIN_CHARGE, MIN_EFFICIENCY, MIN_CAPACITY
from ...model.state import State, StorageState


class Battery(Device):
    """Base electricity storage class."""
    def __init__(self, storage_params: StorageParams):
        self._state_of_charge = storage_params.get("initial_charge", NO_CHARGE)
        self._charging_efficiency = storage_params.get("charging_efficiency", DEFAULT_EFFICIENCY)
        self._discharging_efficiency = storage_params.get("discharging_efficiency", DEFAULT_EFFICIENCY)
        self._energy_capacity = storage_params.get("energy_capacity", MAX_CAPACITY)
        self._power_capacity = storage_params.get("power_capacity", MAX_CAPACITY)
        self.init_time = storage_params.get("initial_time", INITIAL_TIME)
        init_state = StorageState(state_of_charge=self._state_of_charge, charging_efficiency=self._charging_efficiency, discharging_efficiency=self._discharging_efficiency, power_capacity=self._power_capacity, energy_capacity=self._energy_capacity)
        
        super().__init__(storage_params, init_state=init_state)
        self.action_type = StorageAction
        

    @property
    def power_capacity(self) -> float:
        r"""Maximum amount of power the storage device can store in [kW]."""
        return self._power_capacity
    
    @power_capacity.setter
    def power_capacity(self, power_capacity: float):
        power_capacity = MAX_CAPACITY if power_capacity is None else power_capacity
        assert power_capacity >= MIN_CAPACITY, 'power_capacity must be >= 0.'
        self._power_capacity = power_capacity

    @property
    def energy_capacity(self) -> float:
        r"""Maximum amount of energy the storage device can store in [kWh]."""
        return self._energy_capacity
    
    @energy_capacity.setter
    def energy_capacity(self, energy_capacity: float):
        energy_capacity = MAX_CAPACITY if energy_capacity is None else energy_capacity
        assert energy_capacity >= MIN_CAPACITY, 'energy_capacity must be >= 0.'
        self._energy_capacity = energy_capacity

    @property
    def charging_efficiency(self) -> float:
        r"""Technical efficiency of the charging process."""
        return self._charging_efficiency
    
    @charging_efficiency.setter
    def charging_efficiency(self, charging_efficiency: float):
        charging_efficiency = DEFAULT_EFFICIENCY if charging_efficiency is None else charging_efficiency
        assert charging_efficiency > MIN_EFFICIENCY and charging_efficiency < MAX_EFFICIENCY, 'charging_efficiency must be between 0 and 1.'
        self._charging_efficiency = charging_efficiency

    @property
    def discharging_efficiency(self) -> float:
        r"""Technical efficiency of the discharging process."""
        return self._discharging_efficiency
    
    @discharging_efficiency.setter
    def discharging_efficiency(self, discharging_efficiency: float):
        discharging_efficiency = DEFAULT_EFFICIENCY if discharging_efficiency is None else discharging_efficiency
        assert discharging_efficiency > MIN_EFFICIENCY and discharging_efficiency < MAX_EFFICIENCY, 'discharging_efficiency must be between 0 and 1.'
        self._discharging_efficiency = discharging_efficiency

    @property
    def state_of_charge(self):
        r"""Current state of charge of the storage device."""
        return self._state_of_charge
    
    @state_of_charge.setter
    def state_of_charge(self, state_of_charge: float):
        assert state_of_charge >= MIN_CHARGE, 'state_of_charge must be >= MIN_CHARGE.'
        assert state_of_charge <= self.energy_capacity, 'state_of_charge must be <= capacity.'
        self._state_of_charge = state_of_charge

    def reset(self):
        """Reset `Battery` to initial state."""
        self._power_capacity = self.init_state.power_capacity
        self._energy_capacity = self.init_state.energy_capacity
        self._state_of_charge = self.init_state.state_of_charge
        super().reset()

    
    def update_state(self, state: StorageState) -> None:
        self.energy_capacity = state.energy_capacity
        self.power_capacity = state.power_capacity
        self.state_of_charge = state.state_of_charge
        self.charging_efficiency = state.charging_efficiency
        self.discharging_efficiency = state.discharging_efficiency
        self.current_time = state.current_time
        super().update_state(state)

 
    def get_action_space(self) -> Bounds:
        low = -self.state_of_charge if self.state_of_charge > MIN_CHARGE else MIN_CHARGE
        return Bounds(low=low, high=(self.energy_capacity - self.state_of_charge), shape=(1,), dtype=np.float32)  

    def get_observation_space(self) -> Bounds:
        low = np.array([INITIAL_TIME, MIN_CHARGE, MIN_EFFICIENCY, MIN_EFFICIENCY, MIN_CAPACITY, MIN_CAPACITY])
        high = np.array([MAX_TIME, self.energy_capacity, MAX_EFFICIENCY, MAX_EFFICIENCY, MAX_CAPACITY, MAX_CAPACITY])
        return Bounds(low=low, high=high, shape=(len(low),), dtype=np.float32)
    




        
        




    

        

