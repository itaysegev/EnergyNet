"""
Energy Dynamics Module for Smart Grid Simulation.

This module defines various energy dynamics classes that model the behavior of different
energy devices within the smart grid system. It utilizes abstract base classes to enforce
the implementation of essential methods in derived classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from .energy_dynamcis import EnergyDynamics  # Corrected import path
from ..model.state import GridState
from ..utils.utils import move_time_tick, hourly_pricing


class GridDynamics(EnergyDynamics):
    """
    Grid Energy Dynamics.

    This class models the dynamics of a grid-connected device, managing electricity prices
    and simulating time progression within the grid environment.
    """

    def __init__(self, dynamics_params: Optional[Any] = None) -> None:
        """
        Initialize the GridDynamics instance.

        Parameters
        ----------
        dynamics_params : Optional[Any], default=None
            Parameters defining the grid dynamics behavior. Currently not utilized,
            but can be extended for future enhancements.
        """
        super().__init__(dynamics_params)

    def do(self, action: ArrayLike, state: GridState) -> GridState:
        """
        Execute an action to update the grid state.

        This method updates the electricity price based on the current hour and advances
        the time by one tick.

        Parameters
        ----------
        action : ArrayLike
            The action to perform. For GridDynamics, this could represent pricing adjustments
            or other grid-related actions.
        state : GridState
            The current state of the grid before performing the action.

        Returns
        -------
        GridState
            The updated state of the grid after performing the action.
        """
        # Create a copy of the current state to avoid mutating the original state
        new_state = state.copy()

        # Update the price based on the current hour using the hourly_pricing utility
        new_state.price = hourly_pricing(state.hour)

        # Advance the time by one tick using the move_time_tick utility
        new_state.current_time_step, new_state.hour = move_time_tick(state.current_time_step, state.hour)

        return new_state

    def predict(self, action: ArrayLike, state: Optional[GridState] = None, params: Optional[Any] = None) -> GridState:
        """
        Predict the next grid state based on the given action and current state.

        Parameters
        ----------
        action : ArrayLike
            The action to perform. For GridDynamics, this could represent pricing adjustments
            or other grid-related actions.
        state : Optional[GridState], default=None
            The current state of the grid before performing the action. If `None`, uses the
            default initial state.
        params : Optional[Any], default=None
            Additional parameters for the prediction. Currently not utilized.

        Returns
        -------
        GridState
            The predicted state of the grid after performing the action.
        """
        if state is None:
            raise ValueError("State must be provided for prediction.")

        # Predict the next state without mutating the original state
        predicted_state = state.copy()

        # Predict the price based on the current hour
        predicted_state.price = hourly_pricing(state.hour)

        # Predict the next time step and hour
        predicted_state.current_time_step, predicted_state.hour = move_time_tick(state.current_time_step, state.hour)

        return predicted_state
