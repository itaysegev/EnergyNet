
import numpy as np
from gymnasium.spaces import Box
from typing import List

from ..defs import Bounds






def observation_seperator(observation:dict[str, np.ndarray]):
    """
    Seperates the observation into the agents's observation.

    Parameters:
    observation (dict): The observation of all agents.
    agents (str): The agents to get the observation for.

    Returns:
    dict: The observation of the agents.
    """

    return [observation[name] for name in observation.keys()]


def bounds_to_gym_box(bounds: Bounds) -> Box:
  return Box(
        low=bounds['low'],
        high=bounds['high'],
        shape=bounds['shape'],
        dtype=bounds['dtype']
    )





