"""
Energy Dynamics Module for Smart Grid Simulation.

This module defines various energy dynamics classes that model the behavior of different
energy devices within the smart grid system. It utilizes abstract base classes to enforce
the implementation of essential methods in derived classes.
"""

from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from functools import partial

from ...config import DEFAULT_PRODUCTION
from ..energy_dynamcis import ProductionDynamics  # Corrected import path
from ...model.action import ProduceAction
from ...model.state import ProductionState
from ...data.data import TimeSeriesData
from ...utils.utils import move_time_tick, convert_hour_to_int

NUM_SAMPLES_PER_DAY = 48  # Assuming 30-minute intervals

class PVDynamics(ProductionDynamics):
    """
    Photovoltaic (PV) Energy Dynamics.

    This class models the dynamics of a photovoltaic system within the smart grid,
    handling solar generation based on historical data and managing production capabilities.
    """

    def __init__(self, file_name: str, value_row_name: str, time_row_name: str) -> None:
        """
        Initialize the PVDynamics instance.

        Parameters
        ----------
        file_name : str
            Path to the CSV file containing solar generation data.
        value_row_name : str
            Column name in the CSV file that contains solar generation values.
        time_row_name : str
            Column name in the CSV file that contains time stamps.
        """
        super().__init__()
        self.data = TimeSeriesData(file_name)
        self.solar_data = self.data.get_column(value_row_name)
        self.time_data = self.data.get_column(time_row_name).apply(convert_hour_to_int)
        self.max_production = self.solar_data.max()
        self.init_production = self.solar_data.iloc[0]
        self.current_day_start_idx: Optional[int] = None
        self.num_samples_per_day = NUM_SAMPLES_PER_DAY  # Assuming 30-minute intervals

    def do(
        self,
        action: Union[ArrayLike, ProduceAction],
        state: Optional[ProductionState] = None,
        **kwargs: Any
    ) -> ProductionState:
        """
        Perform an action to update the production state based on solar generation.

        This method updates the solar production based on the current time step, applies
        any production adjustments from the action, and advances the time.

        Parameters
        ----------
        action : Union[ArrayLike, ProduceAction]
            The action to perform. Must be either a NumPy array with a single value
            or a `ProduceAction` instance.
        state : Optional[ProductionState], default=None
            The current state of production. If `None`, a default state must be provided.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        ProductionState
            The updated production state after performing the action.

        Raises
        ------
        ValueError
            If the `action` is invalid or `state` is not provided.
        TypeError
            If the `action` is not of the expected type.
        """
        if state is None:
            raise ValueError("Production state must be provided.")

        # Validate and process action
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError("Action array must contain exactly one value.")
            action_value = ProduceAction.from_numpy(action)
        elif isinstance(action, ProduceAction):
            action_value = action
        else:
            raise TypeError("Action must be a numpy array or ProduceAction instance.")

        
        # Determine the current day index
        if state.current_time_step % self.num_samples_per_day == 0:
            num_days = len(self.solar_data) // self.num_samples_per_day
            if num_days == 0:
                raise ValueError("Insufficient data for the number of samples per day.")
            random_day = np.random.randint(0, num_days)
            self.current_day_start_idx = random_day * self.num_samples_per_day

        if self.current_day_start_idx is None:
            raise ValueError("Current day start index has not been initialized.")

        # Calculate the current index
        idx = self.current_day_start_idx + (state.current_time_step % self.num_samples_per_day)
        solar_generation = self.solar_data.iloc[idx]

        # Validate solar data
        if not isinstance(solar_generation, (float, int)):
            raise TypeError("Solar data must be a numeric value.")
        if solar_generation < 0:
            raise ValueError("Solar generation cannot be negative.")
        if solar_generation > self.max_production:
            raise ValueError("Solar generation exceeds maximum production capacity.")

        # Update production
        new_state = state.copy()
        new_state.production = solar_generation 
        new_state.production = min(new_state.production, self.max_production)  # Cap production

        # Apply natural decay or other dynamics if necessary
        # Placeholder for additional dynamics

        # Advance time
        new_state.current_time_step, new_state.hour = move_time_tick(state.current_time_step, state.hour)

        return new_state

    def predict(
        self,
        action: Union[ArrayLike, ProduceAction],
        state: Optional[ProductionState] = None,
        **kwargs: Any
    ) -> ProductionState:
        """
        Predict the next production state based on the given action and current state.

        This method simulates the production output without mutating the original state.

        Parameters
        ----------
        action : Union[ArrayLike, ProduceAction]
            The action to perform. Must be either a NumPy array with a single value
            or a `ProduceAction` instance.
        state : Optional[ProductionState], default=None
            The current state of production. If `None`, a default state must be provided.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        ProductionState
            The predicted production state after performing the action.

        Raises
        ------
        ValueError
            If the `action` is invalid or `state` is not provided.
        TypeError
            If the `action` is not of the expected type.
        """
        if state is None:
            raise ValueError("Production state must be provided for prediction.")

        # Validate and process action
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError("Action array must contain exactly one value.")
            action_value = ProduceAction.from_numpy(action)
        elif isinstance(action, ProduceAction):
            action_value = action
        else:
            raise TypeError("Action must be a numpy array or ProduceAction instance.")

        production_adjustment = action_value.produce

        # Determine the current day index
        if state.current_time_step % self.num_samples_per_day == 0:
            num_days = len(self.solar_data) // self.num_samples_per_day
            if num_days == 0:
                raise ValueError("Insufficient data for the number of samples per day.")
            random_day = np.random.randint(0, num_days)
            current_day_start_idx = random_day * self.num_samples_per_day
        else:
            if self.current_day_start_idx is None:
                raise ValueError("Current day start index has not been initialized.")
            current_day_start_idx = self.current_day_start_idx

        # Calculate the current index
        idx = current_day_start_idx + (state.current_time_step % self.num_samples_per_day)
        solar_generation = self.solar_data.iloc[idx]

        # Validate solar data
        if not isinstance(solar_generation, (float, int)):
            raise TypeError("Solar data must be a numeric value.")
        if solar_generation < 0:
            raise ValueError("Solar generation cannot be negative.")
        if solar_generation > self.max_production:
            raise ValueError("Solar generation exceeds maximum production capacity.")

        # Predict production
        predicted_state = state.copy()
        predicted_state.production = solar_generation + production_adjustment
        predicted_state.production = min(predicted_state.production, self.max_production)  # Cap production

        # Apply natural decay or other dynamics if necessary
        # Placeholder for additional dynamics

        # Advance time
        predicted_state.current_time_step, predicted_state.hour = move_time_tick(state.current_time_step, state.hour)

        return predicted_state

    def get_current_production_capability(self) -> float:
        """
        Get the current production capability of the PV system.

        Returns
        -------
        float
            The current production capability in [kW].
        """
        return self.max_production

    def predict_production_capability(self, state: ProductionState) -> float:
        """
        Predict the future production capability based on the current state.

        Parameters
        ----------
        state : ProductionState
            The current state of production.

        Returns
        -------
        float
            The predicted production capability in [kW].
        """
        # Placeholder for more sophisticated prediction logic
        # For simplicity, return the maximum production
        return self.max_production
