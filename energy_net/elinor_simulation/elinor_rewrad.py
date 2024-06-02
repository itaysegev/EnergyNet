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
        self.A = 5
        self.B = 10

    # Elinor's reward function 
    def calculate(self, curr_state, action, next_state, **kwargs) -> float:
        pg = next_state['consumption'] + action.item()
        return -1 * (self.A * pg ** 2 + self.B) if pg > 0 else 0