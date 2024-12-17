# energy_net_env/rewards/cost_reward.py

from energy_net.rewards.base_reward import BaseReward
from typing import Dict, Any

class CostReward(BaseReward):
    """
    Reward function based on minimizing the net cost of energy transactions.
    """

    def __init__(self):
        """
        Initializes the CostReward with specific pricing.
        """
        

    def compute_reward(self, info: Dict[str, Any]) -> float:
        """
        Computes the reward as the negative net cost.

        Args:
            info (Dict[str, Any]): Contains 'net_exchange' and 'pricing_function'.

        Returns:
            float: Negative net cost.
        """
        buy_amount = info.get('net_exchange', 0.0)
        pricing_function = info.get('pricing_function')
        
        reward = -1 * pricing_function(buy_amount) 
        

        return reward
