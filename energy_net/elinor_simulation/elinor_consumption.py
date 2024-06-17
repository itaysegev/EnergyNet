from ..config import  NO_CONSUMPTION,  PRED_CONST_DUMMY, MIN_POWER
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from datetime import datetime, timedelta
from ..defs import Bounds
from ..model.action import ConsumeAction
from ..elinor_simulation.state import ConsumptionState 
from ..network_entity import  ElementaryNetworkEntity
from ..entities.params import ConsumptionParams


class ElinorUnitConsumption(ElementaryNetworkEntity):
    def __init__(self, consumption_params:ConsumptionParams):
        self._date = None
        self._state = None
        super().__init__(name=consumption_params["name"], energy_dynamics=consumption_params["energy_dynamics"])
        self._init_state = ConsumptionState(consumption=NO_CONSUMPTION, next_consumption=NO_CONSUMPTION)
        self._state = self._init_state
        self.net_load_demand = self.calculate_net_load_demand()
    
    # TODO: Implement the informative predict_next_consumption function 
    def predict_next_consumption(self, param: ConsumptionParams = None) -> float:
        return PRED_CONST_DUMMY

    def step(self, action: ConsumeAction):
        self.date = self._state.promote_date()
        
        
        

    def reset(self) -> ConsumptionState:
        return self._init_state

      
    def get_current_state(self):
        self._state['consumption'] = self.get_current_consumption()
        self._state['next_consumption'] = self.predict_next_consumption()
        return self._state

    #TODO: Implement the get_current_state function according to the consumption function 
    def get_current_consumption(self):
        total_minutes = (self.date - self._init_state['date']).total_seconds() / 60
        time_step = int(total_minutes // 30) % len(self.net_load_demand)
        return self.net_load_demand[time_step]

    def get_observation_space(self):
        low = NO_CONSUMPTION
        high = np.inf
        return Bounds(low=low, high=high, dtype=np.float32)

    def get_action_space(self):
        return Bounds(low=MIN_POWER, high=self.max_electric_power, dtype=np.float32)
    
    @property
    def date(self):
        return self._date


    @date.setter
    def date(self, date):
        if self._date == None:
            self._init_state = ConsumptionState(consumption=NO_CONSUMPTION, next_consumption=NO_CONSUMPTION)
            self._init_state.set_date(date)
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        self._date = date
        if self._state: self._state.set_date(date)

    # Elinor's consumption function
    @staticmethod
    def calculate_net_load_demand():
        df = pd.read_csv('data.csv')

        consumption_data = df.Load.to_numpy()

        consumption_data = [int(x.replace(',', '')) for x in consumption_data]

        return consumption_data
        
        