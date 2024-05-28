from ..config import  NO_CONSUMPTION,  PRED_CONST_DUMMY, MIN_POWER
import numpy as np
from datetime import datetime, timedelta
from ..defs import Bounds
from ..model.action import ConsumeAction
from ..elinor_simulation.state import ConsumptionState 
from ..network_entity import  ElementaryNetworkEntity
from ..entities.params import ConsumptionParams


class ElinorUnitConsumption(ElementaryNetworkEntity):
    def __init__(self, consumption_params:ConsumptionParams):
        super().__init__(name=consumption_params["name"], energy_dynamics=consumption_params["energy_dynamics"])
        date = consumption_params["date"] if "date" in consumption_params else datetime.now()
        self._init_state = ConsumptionState(consumption=NO_CONSUMPTION, next_consumption=self.predict_next_consumption(), date=date)
        self._state = self.reset()

    def predict_next_consumption(self, param: ConsumptionParams = None) -> float:
        return PRED_CONST_DUMMY

    def step(self, action: ConsumeAction):
        # Update the state with the current consumption
        date = self._state.promote_date()
        self._state =  ConsumptionState(consumption=action['consume'], next_consumption=self.predict_next_consumption(), 
                                        date = date)
        print(self._state)
        
        

    def reset(self) -> ConsumptionState:
        return self._init_state

      
    def get_current_state(self):
        return self._state

    #TODO: Implement the get_current_state function according to the consumption function 
    def get_current_consumption(self):
        pass

    def get_observation_space(self):
        low = NO_CONSUMPTION
        high = np.inf
        return Bounds(low=low, high=high, dtype=np.float32)

    def get_action_space(self):
        return Bounds(low=MIN_POWER, high=self.max_electric_power, dtype=np.float32)