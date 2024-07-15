import copy
from ..config import MIN_PRODUCTION, NO_CONSUMPTION, DEFAULT_INIT_POWER, DEFAULT_EFFICIENCY, INITIAL_TIME, MAX_PRODUCTION, MAX_ELECTRIC_POWER
from datetime import datetime, timedelta


class State():
    def __init__(self, current_time:datetime = INITIAL_TIME):
        self.current_time = current_time

    def get_timedelta_state(self, delta_hours):
        timedelta_state = copy.deepcopy(self)
        timedelta_state.current_time += timedelta(hours = delta_hours)
        return timedelta_state


class StorageState(State):
    def __init__(self, current_time:datetime  = INITIAL_TIME, state_of_charge:float = DEFAULT_INIT_POWER, charging_efficiency:float = DEFAULT_EFFICIENCY,discharging_efficiency:float = DEFAULT_EFFICIENCY, power_capacity:float = DEFAULT_INIT_POWER, energy_capacity:float = DEFAULT_INIT_POWER):
        super().__init__(current_time)
        self.state_of_charge=state_of_charge
        self.charging_efficiency=charging_efficiency
        self.discharging_efficiency=discharging_efficiency
        self.power_capacity=power_capacity
        self.energy_capacity=energy_capacity

class ProductionState(State):
    def __init__(self, current_time:datetime =INITIAL_TIME, max_produce:float = MAX_PRODUCTION, production:float = MIN_PRODUCTION):
        super().__init__(current_time)
        self.max_produce = max_produce
        self.production = production

class ConsumptionState(State):
    def __init__(self, current_time:datetime =INITIAL_TIME, max_electric_power:float = MAX_ELECTRIC_POWER, efficiency:float = DEFAULT_EFFICIENCY, consumption:float = NO_CONSUMPTION):
        super().__init__(current_time)
        self.max_electric_power = max_electric_power
        self.efficiency = efficiency
        self.consumption = consumption



