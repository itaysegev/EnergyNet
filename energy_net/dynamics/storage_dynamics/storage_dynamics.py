import numpy as np
from functools import partial

from ..energy_dynamcis import StorageDynamics
from ...model.state import StorageState
from ...model.action import StorageAction
from ...config import MIN_CHARGE, MIN_EXPONENT, MAX_EXPONENT, DEFAULT_LIFETIME_CONSTANT
from ...utils.utils import move_time_tick
import numpy as np
from typing import Union


class BatteryDynamics(StorageDynamics):
    def __init__(self) -> None:
        super().__init__()

    def do(self, action: Union[np.ndarray, StorageAction], state: StorageState=None, **kwargs) -> StorageState:
        """Perform action on battery.
        parameters
        ----------
        action : Numpy array
            Action to be performed. Must be a numpy array with a single value.
        state : StorageState
            Current state of the battery.
        return : StorageState
            New state of charge in [kWh].
        """
        # Check device parameters
        assert state.energy_capacity >= 0, "energy capacity must be greater than zero."
        assert 0 <= state.charging_efficiency <= 1, "charging efficiency must be between 0 and 1."
        assert 0 <= state.discharging_efficiency <= 1, "discharging_efficiency efficiency must be between 0 and 1."

        action_value = StorageAction.from_numpy(action) if isinstance(action, np.ndarray) else action
        value = action_value.charge
        
        # Charging and discharging losses
        if value > 0:  # Charge
            value *= state.charging_efficiency
        else:  # Discharge
            value *= state.discharging_efficiency

        # Natural decay losses
        lifetime_constant = kwargs.get('lifetime_constant', DEFAULT_LIFETIME_CONSTANT)
        
        if value is not None:
            new_state = state.copy()
            if value > MIN_CHARGE:  # Charge
                new_state.state_of_charge = min(state.state_of_charge + value, state.energy_capacity)
            else:  # Discharge
                new_state.state_of_charge = max(state.state_of_charge + value, MIN_CHARGE)

            exp_mult = partial(self.exp_mult, state=state, lifetime_constant=lifetime_constant)
            new_state.energy_capacity = exp_mult(state.energy_capacity)
            new_state.power_capacity = exp_mult(state.power_capacity)
            new_state.charging_efficiency = state.charging_efficiency
            new_state.discharging_efficiency = state.discharging_efficiency
            new_state.current_time_step, new_state.hour = move_time_tick(state.current_time_step, state.hour)
            return new_state
        else:
            raise ValueError('Invalid action')

    def predict(self, action: StorageAction, state: StorageState=None, **kwargs):
        """Predict the next state of the battery given an action.
        """
        # Check device parameters
        assert state.energy_capacity >= 0, "energy capacity must be greater than zero."
        assert 0 <= state.charging_efficiency <= 1, "charging efficiency must be between 0 and 1."
        assert 0 <= state.discharging_efficiency <= 1, "discharging_efficiency efficiency must be between 0 and 1."

        value = action.charge

        # Charging and discharging losses
        if value > 0:  # Charge
            value *= state.charging_efficiency
        else:  # Discharge
            value *= state.discharging_efficiency

        # Natural decay losses
        lifetime_constant = kwargs.get('lifetime_constant', DEFAULT_LIFETIME_CONSTANT)

        if value is not None:
            new_state = state.copy()
            if value > MIN_CHARGE:  # Charge
                new_state.state_of_charge = min(state.state_of_charge + value, state.energy_capacity)
            else:  # Discharge
                new_state.state_of_charge = max(state.state_of_charge + value, MIN_CHARGE)

            exp_mult = partial(self.exp_mult, state=state, lifetime_constant=lifetime_constant)
            new_state.energy_capacity = exp_mult(state.energy_capacity)
            new_state.power_capacity = exp_mult(state.power_capacity)
            new_state.charging_efficiency = state.charging_efficiency
            new_state.discharging_efficiency = state.discharging_efficiency
            new_state.current_time_step = move_time_tick(new_state.current_time_step)
            return new_state
        else:
            raise ValueError('Invalid action')

    @staticmethod
    def exp_mult(x, state, lifetime_constant):
        if lifetime_constant == 0:
            return x  # or handle the zero division case in another way
        else:
            # Clamp the exponent value to prevent overflow
            exponent = state.current_time_step / float(lifetime_constant)
            exponent = max(MIN_EXPONENT, min(MAX_EXPONENT, exponent))
            return x * np.exp(-exponent)

        

    

