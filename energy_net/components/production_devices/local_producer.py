import numpy as np
from typing import Any

from ..device import Device
from ..params import ProductionParams
from ...defs import Bounds
from ...model.state import ProductionState
from ...model.action import ProduceAction
from ...config import (
    MIN_POWER,
    MIN_PRODUCTION,
    INITIAL_HOUR,
    MAX_HOUR,
    MAX_ELECTRIC_POWER,
    DEFAULT_SELF_CONSUMPTION,
    INITIAL_TIME,
    MAX_TIME,
)


class PrivateProducer(Device):
    """
    Represents a private producer device in the smart grid system.

    This class models a producer that can generate electric power and manage its production parameters.
    It utilizes production parameters to initialize its state and action space.

    Parameters
    ----------
    production_params : ProductionParams
        Parameters defining the production behavior of the device.

    Attributes
    ----------
    max_production : float
        Maximum electric power the device can produce.
    init_max_production : float
        Initial maximum production, used for resetting the device.
    production : float
        Current production level of the device.
    self_consumption : float
        Amount of produced power consumed by the device itself.
    action_type : ProduceAction
        The type of action the device can perform.
    """

    def __init__(self, production_params: ProductionParams):
        self.max_production = production_params["max_production"]
        self.init_max_production = self.max_production
        self.production = production_params["init_production"]
        self.self_consumption = (
            production_params["self_consumption"]
            if "self_consumption" in production_params
            else DEFAULT_SELF_CONSUMPTION
        )
        self.action_type = ProduceAction

        init_state = ProductionState(
            max_production=self.max_production,
            production=self.production,
        )
        super().__init__(production_params, init_state=init_state)

    @property
    def current_state(self) -> ProductionState:
        """
        Get the current state of the device.

        Returns
        -------
        ProductionState
            The current production state of the device.
        """
        return ProductionState(
            max_production=self.max_production,
            production=self.production,
        )

    @property
    def max_production(self) -> float:
        """float: Maximum electric power the device can produce."""
        return self._max_production

    @max_production.setter
    def max_production(self, max_production: float):
        """
        Set the maximum electric power.

        Parameters
        ----------
        max_production : float
            The new maximum electric power value.

        Raises
        ------
        AssertionError
            If `max_production` is less than `MIN_POWER`.
        """
        assert (
            max_production >= MIN_POWER
        ), f"max_production must be >= {MIN_POWER}."
        self._max_production = max_production

    @property
    def production(self) -> float:
        """float: Current production level of the device."""
        return self._production

    @production.setter
    def production(self, production: float):
        """
        Set the current production level.

        Parameters
        ----------
        production : float
            The new production level.

        Raises
        ------
        AssertionError
            If `production` is not within [MIN_PRODUCTION, max_production].
        """
        assert (
            MIN_PRODUCTION <= production <= self.max_production
        ), f"production must be in [{MIN_PRODUCTION}, {self.max_production}]."
        self._production = production

    def get_current_state(self) -> ProductionState:
        """
        Retrieve the current state of the device.

        Returns
        -------
        ProductionState
            The current production state.
        """
        return self.current_state

    def get_reward(self) -> float:
        """
        Calculate and return the reward for the current state.

        Returns
        -------
        float
            The self-consumption value as the reward.
        """
        return self.self_consumption

    def reset(self) -> ProductionState:
        """
        Reset the device to its initial state.

        Returns
        -------
        ProductionState
            The state after resetting the device.
        """
        super().reset()
        self.max_production = self.init_max_production
        self.production = MIN_PRODUCTION  # Assuming reset production to minimum
        self.self_consumption = DEFAULT_SELF_CONSUMPTION
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
            high=self.max_production,
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
        # Define the lower and upper bounds for each dimension of the observation space
        low = np.array(
            [INITIAL_TIME, INITIAL_HOUR, MIN_POWER, MIN_PRODUCTION],
            dtype=np.float32,
        )  # Lower bounds for observations
        high = np.array(
            [MAX_TIME, MAX_HOUR, MAX_ELECTRIC_POWER, self.max_production],
            dtype=np.float32,
        )  # Upper bounds for observations
        return Bounds(
            low=low,
            high=high,
            shape=(len(low),),
            dtype=np.float32,
        )
