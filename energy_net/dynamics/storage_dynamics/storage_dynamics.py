import numpy as np
from functools import partial

from ..energy_dynamcis import StorageDynamics
from ...model.state import StorageState
from ...model.action import StorageAction, EnergyAction
from ...config import MIN_CHARGE, MIN_EXPONENT, MAX_EXPONENT, DEFAULT_LIFETIME_CONSTANT
from ...utils.utils import move_time_tick


class BatteryDynamics(StorageDynamics):
    def __init__(self) -> None:
        super().__init__()

    def do(self, action: EnergyAction, state: StorageState=None, params= None) -> StorageState:
        
        """Perform action on battery.
            parameters
            ----------
            action : Numpy array
                Action to be performed. Must be a numpy array with a single value.
            state : BatteryState
                Current state of the battery.
            lifetime_constant : float
            return : BatteryState
                New state of charge in [kWh].
        """

        value = action["charge"] if isinstance(action, dict) else action
        lifetime_constant = DEFAULT_LIFETIME_CONSTANT
        if params and 'lifetime_constant' in params:
            lifetime_constant = params.get('lifetime_constant')
        if value is not None:
            new_state = state.copy()
            if value > MIN_CHARGE: # Charge
                new_state['state_of_charge'] = min(state['state_of_charge'] + value, state['energy_capacity'])
            else: # Discharge
                new_state['state_of_charge'] = max(state['state_of_charge'] + value, MIN_CHARGE)

            exp_mult = partial(self.exp_mult, state=state, lifetime_constant=lifetime_constant)
            new_state['energy_capacity'] = exp_mult(state['energy_capacity'])
            new_state['power_capacity'] = exp_mult(state['power_capacity'])
            new_state['charging_efficiency'] = exp_mult(state['charging_efficiency'])
            new_state['discharging_efficiency'] = exp_mult(state['discharging_efficiency'])
            new_state['current_time'] = move_time_tick(new_state['current_time'])
            return new_state	
        else:
            raise ValueError('Invalid action')

    def predict(self, action: EnergyAction, state: StorageState=None, params= None):
        pass
    
    @staticmethod
    def exp_mult(x, state, lifetime_constant):
        if lifetime_constant == 0:
            return x  # or handle the zero division case in another way
        else:
            # Clamp the exponent value to prevent overflow
            exponent = state.current_time / float(lifetime_constant)
            exponent =  max(MIN_EXPONENT, min(MAX_EXPONENT, exponent))
            return x * np.exp(-exponent)
        

    

