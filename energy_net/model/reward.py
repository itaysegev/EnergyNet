from typing import Any, Mapping, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

Reward = List[float]

@dataclass(frozen=True)
class RewardFunction(ABC):
    """
    Abstract base class for reward functions.

    This class defines the interface for all reward functions, ensuring that any
    concrete implementation provides the necessary logic to calculate rewards
    based on the current state, action, and next state of the environment.

    Attributes
    ----------
    env_metadata : Mapping[str, Any]
        General static information about the environment, such as configuration
        parameters, limits, or other relevant data.
    kwargs : dict
        Additional keyword arguments for custom reward calculations.
    """

    env_metadata: Mapping[str, Any]
    kwargs: dict = field(default_factory=dict)

    @abstractmethod
    def calculate(
        self, 
        curr_state: Any, 
        action: Any, 
        next_state: Any, 
        **kwargs: Any
    ) -> float:
        """
        Calculate the reward based on the transition from the current state to the next state.

        Parameters
        ----------
        curr_state : Any
            The current state of the environment before the action is taken.
        action : Any
            The action taken by the agent.
        next_state : Any
            The state of the environment after the action is taken.
        kwargs : Any
            Additional parameters that might influence reward calculation.

        Returns
        -------
        float
            The calculated reward for the transition.
        """
        pass

    def reset(self) -> None:
        """
        Reset any internal variables or state at the start of a new episode.

        This method can be overridden by subclasses to provide specific reset
        logic if necessary.
        """
        pass
