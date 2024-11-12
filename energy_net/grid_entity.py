import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Union, Optional, Any, Callable
import numpy as np

from energy_net.dynamics.energy_dynamcis import EnergyDynamics
from energy_net.utils.utils import AggFunc
from energy_net.model.action import EnergyAction
from energy_net.model.state import State
from energy_net.model.reward import Reward
from energy_net.defs import Bounds


class GridEntity:
    """
    Abstract base class for all grid entities. Defines the interface for stepping through actions,
    predicting outcomes, managing state, and handling rewards.
    """

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            The name of the grid entity.
        """
        self.name = name

    @abstractmethod
    def step(self, actions: Union[np.ndarray, EnergyAction], **kwargs) -> None:
        """
        Execute the given action(s) and update the entity's state.

        Parameters
        ----------
        actions : Union[np.ndarray, EnergyAction]
            The action(s) to perform.
        kwargs : Any
            Additional keyword arguments.
        """
        pass

    @abstractmethod
    def predict(self, action: EnergyAction, state: State) -> Reward:
        """
        Predict the outcome of performing the given action on the given state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : State
            The current state.

        Returns
        -------
        Reward
            The predicted reward after performing the action.
        """
        pass

    @abstractmethod
    def update_system_state(self) -> None:
        """
        Update the system's state based on the current dynamics.
        """
        pass

    @abstractmethod
    def get_state(self) -> State:
        """
        Retrieve the current state of the grid entity.

        Returns
        -------
        State
            The current state.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the grid entity to its initial state.
        """
        pass

    @abstractmethod
    def get_observation_space(self) -> Bounds:
        """
        Get the observation space bounds for the entity.

        Returns
        -------
        Bounds
            The observation space bounds.
        """
        pass

    @abstractmethod
    def get_action_space(self) -> Bounds:
        """
        Get the action space bounds for the entity.

        Returns
        -------
        Bounds
            The action space bounds.
        """
        pass


class ElementaryGridEntity(GridEntity):
    """
    Represents a basic grid entity with its own state and energy dynamics.
    """

    def __init__(self, name: str, energy_dynamics: EnergyDynamics, init_state: State):
        """
        Initialize the ElementaryGridEntity.

        Parameters
        ----------
        name : str
            The name of the grid entity.
        energy_dynamics : EnergyDynamics
            The energy dynamics governing the entity.
        init_state : State
            The initial state of the entity.
        """
        super().__init__(name)
        self.energy_dynamics = energy_dynamics
        self.state = copy.deepcopy(init_state)
        self.init_state = copy.deepcopy(init_state)

    def step(self, action: Union[np.ndarray, EnergyAction], **kwargs: Any) -> None:
        """
        Execute the given action and update the state.

        Parameters
        ----------
        action : Union[np.ndarray, EnergyAction]
            The action to perform.
        kwargs : Any
            Additional keyword arguments.
        """
        if not (isinstance(action, np.ndarray) or isinstance(action, EnergyAction)):
            raise TypeError("Unsupported action type.")

        new_state = self.energy_dynamics.do(action=action, state=self.state, **kwargs)
        self.state = new_state

    def predict(self, action: EnergyAction, state: State) -> Reward:
        """
        Predict the reward for performing the given action on the given state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : State
            The current state.

        Returns
        -------
        Reward
            The predicted reward.
        """
        predicted_state = self.energy_dynamics.predict(action=action, state=state)
        # Assuming Reward can be derived from the predicted state
        return self.calculate_reward(predicted_state)

    def calculate_reward(self, state: State) -> Reward:
        """
        Calculate the reward based on the state.

        Parameters
        ----------
        state : State
            The state to evaluate.

        Returns
        -------
        Reward
            The calculated reward.
        """
        # Placeholder for reward calculation logic
        return Reward()

    def update_system_state(self) -> None:
        """
        Update the system's state based on energy dynamics.
        """
        # Placeholder for state update logic
        pass

    def get_state(self) -> State:
        """
        Retrieve the current state of the entity.

        Returns
        -------
        State
            The current state.
        """
        return self.state

    def reset(self) -> None:
        """
        Reset the entity to its initial state.
        """
        self.state = copy.deepcopy(self.init_state)

    def get_observation_space(self) -> Bounds:
        """
        Get the observation space bounds for the entity.

        Returns
        -------
        Bounds
            The observation space bounds.
        """
        # Placeholder for observation space definition
        low = np.array([0.0])  # Example low bounds
        high = np.array([100.0])  # Example high bounds
        return Bounds(low=low, high=high, dtype=float)

    def get_action_space(self) -> Bounds:
        """
        Get the action space bounds for the entity.

        Returns
        -------
        Bounds
            The action space bounds.
        """
        # Placeholder for action space definition
        low = np.array([-10.0])  # Example low bounds
        high = np.array([10.0])  # Example high bounds
        return Bounds(low=low, high=high, dtype=float)



class CompositeGridEntity(GridEntity):
    """ 
    This class is a composite grid entity that is composed of other grid entities. It provides an interface for stepping through actions,
    predicting the outcome of actions, getting the current state, updating the state, and getting the reward.
    Represents a composite grid entity composed of multiple sub-entities.
    Manages actions and predictions across all sub-entities and aggregates rewards.
    """
    def __init__(
        self,
        name: str,
        sub_entities: Optional[Dict[str, GridEntity]]= None,
        agg_func: Optional[AggFunc] = None
    ):
        super().__init__(name)
        self.sub_entities: Dict[str,GridEntity] = OrderedDict(sub_entities) if sub_entities else OrderedDict()
        self.agg_func = agg_func

    def step(self, actions: Union[np.ndarray, Dict[str, EnergyAction]], **kwargs: Any) -> None:
        """
        Execute actions on all sub-entities.

        Parameters
        ----------
        actions : Union[np.ndarray, Dict[str, EnergyAction]]
            The actions to perform. Can be a NumPy array (ordered) or a dictionary mapping sub-entity names to actions.
        kwargs : Any
            Additional keyword arguments.
        
        Raises
        ------
        ValueError
            If the number of actions does not match the number of sub-entities when actions are provided as an array.
        TypeError
            If actions are neither a NumPy array nor a dictionary.
        """
        if isinstance(actions, np.ndarray):
            if len(actions) != len(self.sub_entities):
                raise ValueError(
                    f"Number of actions ({len(actions)}) does not match number of sub-entities ({len(self.sub_entities)})."
                )
            for (entity_name, entity), action in zip(self.sub_entities.items(), actions):
                entity.step(action, **kwargs)
        elif isinstance(actions, dict):
            for entity_name, action in actions.items():
                if entity_name not in self.sub_entities:
                    raise ValueError(f"Sub-entity '{entity_name}' does not exist in CompositeNetworkEntity.")
                self.sub_entities[entity_name].step(action, **kwargs)
        else:
            raise TypeError("Actions must be either a NumPy array or a dictionary of EnergyAction instances.")

    def predict(self, actions: Union[np.ndarray, Dict[str, EnergyAction]]) -> Union[Reward, Dict[str, Reward]]:
        """
        Predict the outcome of performing the given actions on all sub-entities.

        Parameters
        ----------
        actions : Union[np.ndarray, Dict[str, EnergyAction]]
            The actions to predict. Can be a NumPy array (ordered) or a dictionary mapping sub-entity names to actions.

        Returns
        -------
        Union[Reward, Dict[str, Reward]]
            The aggregated reward if agg_func is provided, else a dictionary mapping sub-entity names to their predicted rewards.
        
        Raises
        ------
        ValueError
            If the number of actions does not match the number of sub-entities when actions are provided as an array.
        TypeError
            If actions are neither a NumPy array nor a dictionary.
        """
        predicted_rewards: Dict[str, Reward] = {}

        if isinstance(actions, np.ndarray):
            if len(actions) != len(self.sub_entities):
                raise ValueError(
                    f"Number of actions ({len(actions)}) does not match number of sub-entities ({len(self.sub_entities)})."
                )
            for (entity_name, entity), action in zip(self.sub_entities.items(), actions):
                predicted_rewards[entity_name] = entity.predict(action, entity.get_state())
        elif isinstance(actions, dict):
            for entity_name, action in actions.items():
                if entity_name not in self.sub_entities:
                    raise ValueError(f"Sub-entity '{entity_name}' does not exist in CompositeNetworkEntity.")
                predicted_rewards[entity_name] = self.sub_entities[entity_name].predict(action, self.sub_entities[entity_name].get_state())
        else:
            raise TypeError("Actions must be either a NumPy array or a dictionary of EnergyAction instances.")

        if self.agg_func:
            aggregated_reward = self.agg_func(predicted_rewards)
            return aggregated_reward
        else:
            return predicted_rewards

    def update_system_state(self) -> None:
        """
        Update the system's state by updating all sub-entities.
        """
        for entity in self.sub_entities.values():
            entity.update_system_state()

    def get_state(self, numpy_arr: bool = False) -> Union[Dict[str, State], np.ndarray, Reward]:
        """
        Retrieve the current state of all sub-entities.

        Parameters
        ----------
        numpy_arr : bool, optional
            If True, returns the state as a concatenated NumPy array, by default False.

        Returns
        -------
        Union[Dict[str, State], np.ndarray, Reward]
            The aggregated state as a dictionary, NumPy array, or aggregated Reward if agg_func is provided.
        """
        states: Dict[str, State] = {entity_name: entity.get_state() for entity_name, entity in self.sub_entities.items()}

        if self.agg_func:
            aggregated_state = self.agg_func(states)
            if numpy_arr:
                # Assuming aggregated_state can be converted to NumPy
                return aggregated_state.to_numpy()
            return aggregated_state

        if numpy_arr:
            # Concatenate all states into a single NumPy array
            concatenated = np.concatenate([state.to_numpy() for state in states.values()])
            return concatenated

        return states

    def reset(self) -> None:
        """
        Reset all sub-entities to their initial states.
        """
        for entity in self.sub_entities.values():
            entity.reset()

    def get_observation_space(self) -> Dict[str, Bounds]:
        """
        Retrieve the observation space for each sub-entity.

        Returns
        -------
        Dict[str, Bounds]
            A dictionary mapping sub-entity names to their observation bounds.
        """
        return {name: entity.get_observation_space() for name, entity in self.sub_entities.items()}

    def get_action_space(self) -> Dict[str, Bounds]:
        """
        Retrieve the action space for each sub-entity.

        Returns
        -------
        Dict[str, Bounds]
            A dictionary mapping sub-entity names to their action bounds.
        """
        return {name: entity.get_action_space() for name, entity in self.sub_entities.items()}
