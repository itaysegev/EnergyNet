"""
This code is based on https://github.com/intelligent-environments-lab/CityLearn/blob/master/citylearn/energy_model.py
"""

import numpy as np
from typing import Any

from .device import Device
from .params import DeviceParams
from ..model.state import GridState
from ..model.action import ConsumeAction
from ..defs import Bounds
from ..config import (
    MIN_PRICE,
    MAX_PRICE,
    INITIAL_TIME,
    MAX_TIME,
    INITIAL_HOUR,
    MAX_HOUR,
)


class GridDevice(Device):
    """
    Represents a grid-connected device within the smart grid system.

    This class models a device that interacts with the power grid, managing its pricing and energy consumption.
    It utilizes grid-specific parameters to initialize its state and action space.

    Parameters
    ----------
    params : DeviceParams
        Parameters defining the grid device's behavior and configuration.

    Attributes
    ----------
    price : float
        Current price of electricity in the grid.
    action_type : ConsumeAction
        The type of action the device can perform.
    """

    def __init__(self, params: DeviceParams):
        """
        Initialize the GridDevice with given parameters.

        Parameters
        ----------
        params : DeviceParams
            A dictionary containing initialization parameters.
        """
        init_state = GridState(price=MIN_PRICE)
        super().__init__(params, init_state=init_state)
        self.action_type = ConsumeAction  # Define the action type for this device

    def reset(self) -> None:
        """
        Reset the GridDevice to its initial state.

        This method restores the device's properties to their initial values as defined in the initial state.
        """
        super().reset()

    def get_action_space(self) -> Bounds:
        """
        Define the action space for the GridDevice.

        The action space is defined based on the minimum and maximum electricity prices.
        It allows actions to adjust the device's interaction with the grid within feasible price limits.

        Returns
        -------
        Bounds
            The bounds of the action space, specifying the range of possible price adjustments.
        """
        return Bounds(
            low=MIN_PRICE,
            high=MAX_PRICE,
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation_space(self) -> Bounds:
        """
        Define the observation space for the GridDevice.

        The observation space includes time, hour, and current electricity price.

        Returns
        -------
        Bounds
            The bounds of the observation space, specifying the range of possible observations.
        """
        # Define the lower and upper bounds for each dimension of the observation space
        low = np.array(
            [INITIAL_TIME, INITIAL_HOUR, MIN_PRICE],
            dtype=np.float32,
        )  # Lower bounds for observations
        high = np.array(
            [MAX_TIME, MAX_HOUR, MAX_PRICE],
            dtype=np.float32,
        )  # Upper bounds for observations
        return Bounds(
            low=low,
            high=high,
            shape=(len(low),),
            dtype=np.float32,
        )

    @property
    def current_state(self) -> GridState:
        """
        Get the current state of the GridDevice.

        Returns
        -------
        GridState
            The current grid state of the device.
        """
        return self.state  # Assuming `state` is managed by the parent `Device` class

    def get_reward(self) -> float:
        """
        Calculate and return the reward for the current state.

        This method can be customized to define how rewards are calculated based on the device's state.

        Returns
        -------
        float
            The reward value.
        """
        # Placeholder for reward calculation logic
        # Customize based on specific simulation requirements
        return 0.0
