from typing import Any, Union
import numpy as np
from ..config import INITIAL_TIME, NO_CONSUMPTION, MAX_CONSUMPTION, NO_CHARGE, MAX_CAPACITY
from ..defs import Bounds
from ..model.action import EnergyAction, StorageAction, ConsumeAction
from ..elinor_simulation.state import UnitState as State
from ..network_entity import CompositeNetworkEntity
from ..entities.local_storage import Battery
from ..entities.device import StorageDevice
from ..entities.params import StorageParams, ProductionParams, ConsumptionParams
from ..entities.local_producer import PrivateProducer
from ..elinor_simulation.elinor_consumption import ElinorUnitConsumption

class ElinorUnit(CompositeNetworkEntity):
   
    def __init__(self, name: str, consumption_params_dict:dict[str, ConsumptionParams]=None, storage_params_dict:dict[str,StorageParams]=None, production_params_dict:dict[str,ProductionParams]=None, agg_func=None):

        # holding the elements that should be considered for consumption, production and storage actions
        consumption_dict = {name: ElinorUnitConsumption(params) for name, params in consumption_params_dict.items()}
        self.consumption_keys = list(consumption_dict.keys())
        storage_dict = {name: Battery(storage_params=params, init_time=INITIAL_TIME) for name, params in storage_params_dict.items()}
        self.storage_keys = list(storage_dict.keys())
        production_dict = {name: PrivateProducer(params) for name, params in production_params_dict.items()}
        self.production_keys = list(production_dict.keys())

        # the base class holds the entities in a single array
        sub_entities = {**consumption_dict, **storage_dict, **production_dict}
        super().__init__(name=name, sub_entities=sub_entities, agg_func=agg_func)


        inital_soc = sum(s.state['state_of_charge'] for s in self.get_storage_devices().values())

        self._init_state = State(storage=inital_soc, consumption=NO_CONSUMPTION,
                                  next_consumption=self.predict_next_consumption())
                                  
        self._state = self._init_state

    def perform_joint_action(self, actions:dict[str, EnergyAction]):
        super().step(actions)

    def step(self, action: Union[np.ndarray, StorageAction]):
        # storage action
        action = StorageAction.from_numpy(action) if type(action) == np.ndarray else action
        current_storage = action['charge']
        current_consumption = self._state['consumption']
        # this is how much we buy/sell to the grid
        pg = current_consumption + current_storage

        actions = {self.consumption_keys[0]: ConsumeAction(consume=current_consumption), self.storage_keys[0]: action}
        super().step(actions)


    def update_system_state(self):

        # get (and update) current state
        self._state = self.get_current_state()
        return self._state

        # get current consumption (the load demand that needs to be currently satisfied)
        #cur_comsumption =  self._state['curr_consumption']

        # get current production
        #cur_production = self._state['production']

        # get current price
        #[cur_price_sell, cur_price_buy] = self.get_current_market_price()

        # get storage/trade policy for all entities
        #joint_action = self.get_joint_action(cur_comsumption, cur_production, cur_price_sell, cur_price_buy)

        #return joint_action


    # TODO: this method has to call the agents that decides about the policy
    def get_joint_action(self)->dict[str, EnergyAction]:
        joint_action = {}
        for entitiy_name in self.sub_entities.keys():
            if entitiy_name in self.storage_keys:
                joint_action[entitiy_name] = StorageAction(charge=10)
        return joint_action


    def get_current_market_price(self):
        # todo: remove
        return [8,8]

    def predict(self, actions: Union[np.ndarray, dict[str, Any]]):
        pass

    def predict_next_consumption(self) -> float:
        return sum([self.sub_entities[name].predict_next_consumption() for name in self.consumption_keys])
  

    def get_current_state(self) -> State:
        sum_dict = {}
        for entity_name in self.sub_entities:
            cur_state = self.sub_entities[entity_name].get_current_state()
            for k, v in cur_state.items():
                if v is None or type(v) is not int:
                    v = 0
                sum_dict[k] = sum_dict.get(k, 0) + v
        return State(storage=sum_dict['state_of_charge'], consumption=sum_dict['consumption'], next_consumption=sum_dict['next_consumption'])


    def update_state(self, state: State):
        for entity in self.sub_entities:
            entity.update_state(state[entity.name])

    def get_observation_space(self) -> Bounds:
        low = np.array([NO_CHARGE, NO_CONSUMPTION, NO_CONSUMPTION])
        high = np.array([MAX_CAPACITY, MAX_CONSUMPTION, MAX_CONSUMPTION])
        return Bounds(low=low, high=high, shape=(len(low),) ,dtype=np.float32)


    def get_action_space(self) -> Bounds:
        storage_devices_action_sapces = [v.get_action_space() for v in self.get_storage_devices().values()]
        
        # Combine the Bounds objects into a single Bound object
        combined_low = np.array([bound['low'] for bound in storage_devices_action_sapces])
        combined_high = np.array([bound['high'] for bound in storage_devices_action_sapces])
        return Bounds(low=combined_low, high=combined_high, shape=(len(combined_low),),  dtype=np.float32)


    def reset(self) -> State:
        # return self.apply_func_to_sub_entities(lambda entity: entity.reset())
        self._state = self._init_state
        self._state['next_consumption'] = self.predict_next_consumption()
        for entity in self.sub_entities.values():
            entity.reset()
        return self._state


    def get_storage_devices(self):
        return self.apply_func_to_sub_entities(lambda entity: entity, condition=lambda x: isinstance(x, StorageDevice))
    
    def validate_action(self, actions: dict[str, EnergyAction]):
        for entity_name, action in actions.items():
            if len(action) > 1 or 'charge' not in action.keys():
                raise ValueError(f"Invalid action key {action.keys()} for entity {entity_name}")
            else:
                return True
    # @property
    # def current_storge_state(self):
    #     return sum([s.current_state['state_of_charge'] for s in self.storage_units])

    def apply_func_to_sub_entities(self, func, condition=lambda x: True):
        results = {}
        for name, entity in self.sub_entities.items():
            if condition(entity):
                results[name] = func(entity)
        return results

    def get_next_consumption(self) -> float:
        return sum([self.sub_entities[name].predict_next_consumption() for name in self.consumption_keys])







