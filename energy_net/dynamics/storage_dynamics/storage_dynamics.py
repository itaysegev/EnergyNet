import numpy as np
from functools import partial

from scipy.special import kwargs

from ..energy_dynamcis import StorageDynamics
from ...model.state import StorageState
from ...model.action import StorageAction, EnergyAction
from ...config import MIN_CHARGE, MIN_EXPONENT, MAX_EXPONENT, DEFAULT_LIFETIME_CONSTANT
from ...utils.utils import move_time_tick


class BatteryDynamics(StorageDynamics):
    def __init__(self) -> None:
        super().__init__()

    def do(self, action: EnergyAction, state: StorageState=None, **kwargs) -> StorageState:

        """Perform action on battery.
            parameters
            ----------
            params : Dict
                Holds additional parameters for battery such as lifetime.
            action : Numpy array
                Action to be performed. Must be a numpy array with a single value.
            state : BatteryState
                Current state of the battery.
            lifetime_constant : float
            return : BatteryState
                New state of charge in [kWh].
        """
        # Check device parameters
        assert state['energy_capacity'] >= 0, "energy capacity must be greater than zero."
        assert state['charging_efficiency'] >= 0 and state['charging_efficiency'] <= 1, "charging efficiency must be between 0 and 1."
        assert state['discharging_efficiency'] >= 0 and state['discharging_efficiency'] <= 1, "discharging_efficiency efficiency must be between 0 and 1."

        value = action["charge"] if isinstance(action, dict) else action
        
        # Charging and discharging losses
        if value > 0: # Charge
            value = value * state['charging_efficiency']
        else:
            value = value * state['discharging_efficiency']

        # Natural decay losses
        lifetime_constant = kwargs.get('lifetime_constant', DEFAULT_LIFETIME_CONSTANT)
        
        if value is not None:
            new_state = state.copy()
            if value > MIN_CHARGE:  # Charge
                new_state['state_of_charge'] = min(state['state_of_charge'] + value, state['energy_capacity'])
            else: # Discharge
                new_state['state_of_charge'] = max(state['state_of_charge'] + value, MIN_CHARGE)



            exp_mult = partial(self.exp_mult, state=state, lifetime_constant=lifetime_constant)
            new_state['energy_capacity'] = exp_mult(state['energy_capacity'])
            new_state['power_capacity'] = exp_mult(state['power_capacity'])
            new_state['charging_efficiency'] = state['charging_efficiency']
            new_state['discharging_efficiency'] = state['discharging_efficiency']
            new_state['current_time'] = move_time_tick(new_state['current_time'])
            return new_state	
        else:
            raise ValueError('Invalid action')

    def predict(self, action: EnergyAction, state: StorageState=None, params= None):
        # Check device parameters
        assert state['energy_capacity'] >= 0, "energy capacity must be greater than zero."
        assert state['charging_efficiency'] >= 0 and state[
            'charging_efficiency'] <= 1, "charging efficiency must be between 0 and 1."
        assert state['discharging_efficiency'] >= 0 and state[
            'discharging_efficiency'] <= 1, "discharging_efficiency efficiency must be between 0 and 1."

        value = action["charge"] if isinstance(action, dict) else action

        # Charging and discharging losses
        if value > 0:  # Charge
            value = value * state['charging_efficiency']
        else:
            value = value * state['discharging_efficiency']

        # Natural decay losses
        lifetime_constant = kwargs.get('lifetime_constant', DEFAULT_LIFETIME_CONSTANT)

        if value is not None:
            new_state = state.copy()
            if value > MIN_CHARGE:  # Charge
                new_state['state_of_charge'] = min(state['state_of_charge'] + value, state['energy_capacity'])
            else:  # Discharge
                new_state['state_of_charge'] = max(state['state_of_charge'] + value, MIN_CHARGE)

            exp_mult = partial(self.exp_mult, state=state, lifetime_constant=lifetime_constant)
            new_state['energy_capacity'] = exp_mult(state['energy_capacity'])
            new_state['power_capacity'] = exp_mult(state['power_capacity'])
            new_state['charging_efficiency'] = state['charging_efficiency']
            new_state['discharging_efficiency'] = state['discharging_efficiency']
            new_state['current_time'] = move_time_tick(new_state['current_time'])
            return new_state
        else:
            raise ValueError('Invalid action')

    @staticmethod
    def exp_mult(x, state, lifetime_constant):
        if lifetime_constant == 0:
            return x  # or handle the zero division case in another way
        else:
            # Clamp the exponent value to prevent overflow
            exponent = state['current_time'] / float(lifetime_constant)
            exponent = max(MIN_EXPONENT, min(MAX_EXPONENT, exponent))
            return x * np.exp(-exponent)
        

    

