from ..model.reward import RewardFunction
from typing import Any, Mapping

class ElinorUnitRewardFunction(RewardFunction):
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

    # TODO: implement the reward function
    def calculate(self, curr_state, action, next_state, **kwargs) -> float:
        pass