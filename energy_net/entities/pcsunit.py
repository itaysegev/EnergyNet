from typing import Any, Union, Mapping
import numpy as np
import random
from ..config import INITIAL_TIME, NO_CONSUMPTION, MAX_CONSUMPTION, NO_CHARGE, MAX_CAPACITY, PRED_CONST_DUMMY, MIN_POWER
from ..defs import Bounds
from ..model.action import EnergyAction, StorageAction, TradeAction, ConsumeAction, ProduceAction
from ..model.reward import RewardFunction
from ..model.state import State
from ..network_entity import NetworkEntity, CompositeNetworkEntity, ElementaryNetworkEntity
from ..entities.local_storage import Battery
from ..entities.device import StorageDevice
from ..entities.params import StorageParams, ProductionParams, ConsumptionParams
from ..entities.local_producer import PrivateProducer

class PCSUnit(CompositeNetworkEntity):
    """ A network entity that contains a list of sub-entities. The sub-entities are the devices and the pcsunit itself is the composite entity.
    The PCSUnit entity is responsible for managing the sub-entities and aggregating the reward.
    """
    def __init__(self, name: str, consumption_params_dict:dict[str,ConsumptionParams]=None, storage_params_dict:dict[str,StorageParams]=None, production_params_dict:dict[str,ProductionParams]=None, agg_func=None):

        # holding the elements that should be considered for consumption, production and storage actions
        consumption_dict = {name: PSCUnitConsumption(params) for name, params in consumption_params_dict.items()}
        self.consumption_keys = list(consumption_dict.keys())
        storage_dict = {name: Battery(storage_params=params, init_time=INITIAL_TIME) for name, params in storage_params_dict.items()}
        self.storage_keys = list(storage_dict.keys())
        production_dict = {name: PrivateProducer(params) for name, params in production_params_dict.items()}
        self.production_keys = list(production_dict.keys())

        # the base class holds the entities in a single array
        sub_entities = {**consumption_dict, **storage_dict, **production_dict}
        super().__init__(name=name, sub_entities=sub_entities, agg_func=agg_func)


        inital_soc = sum(s.state['state_of_charge'] for s in self.get_storage_devices().values())
        
        self._init_state = State(storage=inital_soc, curr_consumption=NO_CONSUMPTION, pred_consumption=self.predict_next_consumption())
        self._state = self._init_state

    def perform_joint_action(self, actions:dict[str, EnergyAction]):
        super().step(actions)

    def step(self, action: StorageAction):
        # storage action
        current_storage = action['charge']
        current_consumption = self._state['consumption']
        # this is how much we buy/sell to the grid
        pg = current_consumption+current_storage

        actions = {self.consumption_keys[0]: ConsumeAction(consume=current_consumption),self.storage_keys[0]: action}
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
                if v is None:
                    v = 0
                sum_dict[k] = sum_dict.get(k, 0) + v
        return State(storage=sum_dict['state_of_charge'], curr_consumption=sum_dict['consumption'], pred_consumption=sum_dict['next_consumption'])


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
        self._state['pred_consumption'] = self.predict_next_consumption()
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

class PSCUnitConsumption(ElementaryNetworkEntity):
    def __init__(self, consumption_params:ConsumptionParams):
        super().__init__(name=consumption_params["name"],energy_dynamics=consumption_params["energy_dynamics"])
        self._state = self.reset()

    def predict_next_consumption(self, param: ConsumptionParams = None) -> float:
        return PRED_CONST_DUMMY

    def step(self, action: ConsumeAction):
        # Update the state with the current consumption
        self._state =  State(consumption=action['consume'], next_consumption=self.predict_next_consumption())

    def reset(self) -> State:
        return State(consumption=NO_CONSUMPTION, next_consumption=self.predict_next_consumption())
    
    def get_current_state(self):
        return self._state

    def get_observation_space(self):
        low = NO_CONSUMPTION
        high = np.inf
        return Bounds(low=low, high=high, dtype=np.float32)

    def get_action_space(self):
        return Bounds(low=MIN_POWER, high=self.max_electric_power, dtype=np.float32)


class DefaultPCSUnitRewardFunction(RewardFunction):
    """Dummy reward function class.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    **kwargs : dict
        Other keyword arguments for custom reward calculation.
    """

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs):
        super().__init__(env_metadata, **kwargs)

    def calculate(self, curr_state, action, next_state, **kwargs) -> float:
        # production = curr_state['curr_consumption'] + action.item()
        # time_steps = kwargs.get('time_steps', None)
        # assert time_steps is not None
        # return  -1 * (production if time_steps  == 0 else 2 * production)
        return -1 * action.item()
