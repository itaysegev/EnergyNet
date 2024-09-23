import numpy as np
from typing import Any

from ...defs import Bounds
from ..device import Device
from ...model.state import ConsumptionState
from ...model.action import ConsumeAction
from ...config import (
    MAX_ELECTRIC_POWER,
    MIN_POWER,
    INITIAL_HOUR,
    MAX_HOUR,
    NO_CONSUMPTION,
    INITIAL_TIME,
    MAX_TIME,
)
from ..params import ConsumptionParams


class ConsumerDevice(Device):
    """
    Represents a consumer device in the smart grid system.

    This class models a consumer that can consume electric power from the power grid.
    It uses consumption parameters to initialize its state and action space.

    Parameters
    ----------
    consumption_params : ConsumptionParams
        Parameters defining the consumption behavior of the device.

    Attributes
    ----------
    max_electric_power : float
        Maximum electric power the device can consume.
    init_max_electric_power : float
        Initial maximum electric power, used for resetting the device.
    consumption : float
        Current consumption of the device.
    action_type : ConsumeAction
        The type of action the device can perform.
    """

    def __init__(self, consumption_params: ConsumptionParams):
        self.max_electric_power = consumption_params["max_electric_power"]
        self.init_max_electric_power = self.max_electric_power
        self.consumption = consumption_params["init_consum"]
        self.action_type = ConsumeAction

        init_state = ConsumptionState(
            max_electric_power=self.max_electric_power,
            consumption=self.consumption,
        )
        super().__init__(consumption_params, init_state=init_state)

    @property
    def max_electric_power(self) -> float:
        """float: Maximum electric power the device can consume."""
        return self._max_electric_power

    @max_electric_power.setter
    def max_electric_power(self, max_electric_power: float):
        """
        Set the maximum electric power.

        Parameters
        ----------
        max_electric_power : float
            The new maximum electric power value.

        Raises
        ------
        AssertionError
            If `max_electric_power` is less than `MIN_POWER`.
        """
        assert (
            max_electric_power >= MIN_POWER
        ), f"max_electric_power must be >= {MIN_POWER}."
        self._max_electric_power = max_electric_power

    @property
    def current_state(self) -> ConsumptionState:
        """
        Get the current state of the device.

        Returns
        -------
        ConsumptionState
            The current consumption state of the device.
        """
        return ConsumptionState(
            max_electric_power=self.max_electric_power,
            consumption=self.consumption,
        )

    def reset(self) -> ConsumptionState:
        """
        Reset the device to its initial state.

        Returns
        -------
        ConsumptionState
            The state after resetting the device.
        """
        super().reset()
        self.max_electric_power = self.init_max_electric_power
        self.consumption = NO_CONSUMPTION
        return self.current_state

    def get_action_space(self) -> Bounds:
        """
        Define the action space for the device.

        Returns
        -------
        Bounds
            The bounds of the action space, specifying the range of possible actions.
        """
        return Bounds(
            low=MIN_POWER,
            high=self.max_electric_power,
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation_space(self) -> Bounds:
        """
        Define the observation space for the device.

        Returns
        -------
        Bounds
            The bounds of the observation space, specifying the range of possible observations.
        """
        low = np.array(
            [INITIAL_TIME, INITIAL_HOUR, MIN_POWER, NO_CONSUMPTION],
            dtype=np.float32,
        )  # Lower bounds for observations
        high = np.array(
            [MAX_TIME, MAX_HOUR, MAX_ELECTRIC_POWER, self.max_electric_power],
            dtype=np.float32,
        )  # Upper bounds for observations
        return Bounds(
            low=low,
            high=high,
            shape=(len(low),),
            dtype=np.float32,
        )
