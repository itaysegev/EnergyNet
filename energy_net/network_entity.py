from abc import abstractmethod
from collections import OrderedDict
from typing import Union
import numpy as np

from .dynamics.energy_dynamcis import EnergyDynamics
from .utils.utils import AggFunc
from .model.action import EnergyAction
from .model.state import State
from .model.reward import Reward


class NetworkEntity:
    """
    This is a base class for all network entities. It provides an interface for stepping through actions,
    predicting the outcome of actions, getting the current state, updating the state, and getting the reward.
    """

    def __init__(self, name: str):
        """
        Constructor for the NetworkEntity class.

        Parameters:
        name (str): The name of the network entity.
        """
        self.name = name

    @abstractmethod
    def step(self, actions: dict[str, Union[np.ndarray,EnergyAction]]):
        """
        Perform the given action and return the new state and reward.

        Parameters:
        action (EnergyAction): The action to perform.

        Returns:
        list: The new state and reward after performing the action.
        """
        pass

    @abstractmethod
    def predict(self, action: EnergyAction, state: State):
        """
        Predict the outcome of performing the given action on the given state.

        Parameters:
        action (EnergyAction): The action to perform.
        state (State): The current state.

        Returns:
        list: The predicted new state and reward after performing the action.
        """
        pass

    @abstractmethod
    def update_system_state(self):
        pass


class CompositeNetworkEntity(NetworkEntity):
    """ 
    This class is a composite network entity that is composed of other network entities. It provides an interface for stepping through actions,
    predicting the outcome of actions, getting the current state, updating the state, and getting the reward.
    """

    def __init__(self, name: str, sub_entities: dict[str, NetworkEntity] = None, agg_func: AggFunc = None):
        super().__init__(name)
        self.sub_entities = sub_entities
        self.agg_func = agg_func

    def step(self, actions: dict[str, Union[np.ndarray,EnergyAction]]):
        states = {}
        for entity_name, action in actions.items():
            if type(action) is np.ndarray:
                action = self.sub_entities[entity_name].action_type.from_numpy(action)
            cur_state = self.sub_entities[entity_name].step(action)
            if cur_state:
                states[entity_name] = cur_state
    
        if self.agg_func:
            agg_value = self.agg_func(states)
            return agg_value
        else:
            return states

    def predict(self, actions: Union[np.ndarray, dict[str, EnergyAction]]):

        predicted_states = {}
        if type(actions) is np.ndarray:
            # we convert the entity dict to a list and match action to entities by index
            sub_entities = list(self.sub_entities.values())
            for entity_index, action in enumerate(actions):
                predicted_states[sub_entities[entity_index].name] = sub_entities[entity_index].predict(np.array([action]))

        else:
            for entity_name, action in actions.items():
                predicted_states[entity_name] = self.sub_entities[entity_name].predict(action)

        if self.agg_func:
            agg_value = self.agg_func(predicted_states)
            return agg_value
        else:
            return predicted_states

    def get_joint_action(self)->dict[str, EnergyAction]:
        pass

    def apply_joint_action(self, joint_action:dict[str, EnergyAction]):
            for entity in joint_action:
                # get entity and apply the action
                self.sub_entities[entity].step(joint_action[entity])

class ElementaryNetworkEntity(NetworkEntity):
    """
    This class is an elementary network entity that is composed of other network entities. It provides an interface for stepping through actions,
    predicting the outcome of actions, getting the current state, updating the state, and getting the reward.
    """

    def __init__(self, name, energy_dynamics: EnergyDynamics , init_state:State=None):
        super().__init__(name)
        # if the state is none - this is a stateless entity
        self.state = init_state
        self.init_state = init_state
        self.energy_dynamics = energy_dynamics

    def step(self, action: EnergyAction):
        if self.state:
            new_state = self.energy_dynamics.do(action=action, state=self.state)
            self.update_state(new_state)
            return new_state
        else:
            return self.energy_dynamics.do(action=action)

    def predict(self, action: EnergyAction, state: State):
        predicted_state = self.energy_dynamics.predict(action=action, state=state)
        return predicted_state


    def get_current_state(self) -> State:
        """
        Get the current state of the network entity.

        Returns:
        State: The current state.
        """
        return self.state

    def update_state(self, state: State) -> None:
        self.state = state
        


    def reset(self) -> State:
        self.state = self.init_state
        return self.state

