"""
Energy Dynamics Module for Smart Grid Simulation.

This module defines various energy dynamics classes that model the behavior of different
energy devices within the smart grid system. It utilizes abstract base classes to enforce
the implementation of essential methods in derived classes.
"""

from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike


from ..energy_dynamcis import ConsumptionDynamics  # Corrected import path
from ...model.action import ConsumeAction
from ...model.state import ConsumptionState
from ...data.data import TimeSeriesData
from ...utils.utils import move_time_tick, convert_hour_to_int


class GeneralLoad(ConsumptionDynamics):
    """
    General Load Energy Dynamics.

    This class models the dynamics of a general load within the smart grid,
    handling consumption based on historical data and managing consumption capabilities.
    """

    def __init__(
        self, file_name: str, value_row_name: str, time_row_name: str
    ) -> None:
        """
        Initialize the GeneralLoad instance.

        Parameters
        ----------
        file_name : str
            Path to the CSV file containing load consumption data.
        value_row_name : str
            Column name in the CSV file that contains load consumption values.
        time_row_name : str
            Column name in the CSV file that contains time stamps.
        """
        super().__init__()
        self.data = TimeSeriesData(file_name)
        self.load_data = self.data.get_column(value_row_name)
        self.time_data = self.data.get_column(time_row_name).apply(convert_hour_to_int)
        self.max_electric_power = self.load_data.max()
        self.init_power = self.load_data.iloc[0]
        self.current_day_start_idx: Optional[int] = None
        self.num_samples_per_day = 48  # Assuming 30-minute intervals

    def do(
        self,
        action: Union[ArrayLike, ConsumeAction],
        state: Optional[ConsumptionState] = None,
        **kwargs: Any
    ) -> ConsumptionState:
        """
        Perform an action to update the consumption state based on load data.

        This method updates the load consumption based on the current time step and
        advances the time.

        Parameters
        ----------
        action : Union[ArrayLike, ConsumeAction]
            The action to perform. Must be either a NumPy array with a single value
            or a `ConsumeAction` instance.
        state : Optional[ConsumptionState], default=None
            The current state of consumption. If `None`, a default state must be provided.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        ConsumptionState
            The updated consumption state after performing the action.

        Raises
        ------
        ValueError
            If the `action` is invalid or `state` is not provided.
        TypeError
            If the `action` is not of the expected type.
        """
        if state is None:
            raise ValueError("Consumption state must be provided.")

        # Validate and process action
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError("Action array must contain exactly one value.")
            action_value = ConsumeAction.from_numpy(action)
        elif isinstance(action, ConsumeAction):
            action_value = action
        else:
            raise TypeError("Action must be a numpy array or ConsumeAction instance.")


        # Determine the current day index
        if state.current_time_step % self.num_samples_per_day == 0:
            num_days = len(self.load_data) // self.num_samples_per_day
            if num_days == 0:
                raise ValueError("Insufficient data for the number of samples per day.")
            random_day = np.random.randint(0, num_days)
            self.current_day_start_idx = random_day * self.num_samples_per_day

        if self.current_day_start_idx is None:
            raise ValueError("Current day start index has not been initialized.")

        # Calculate the current index
        idx = self.current_day_start_idx + (state.current_time_step % self.num_samples_per_day)
        load_consumption = self.load_data.iloc[idx]

        # Validate load data
        if not isinstance(load_consumption, (float, int)):
            raise TypeError("Load data must be a numeric value.")
        if load_consumption < 0:
            load_consumption = 0
            # Optionally, log a warning about negative load data being set to zero
        if load_consumption > self.max_electric_power:
            raise ValueError("Load consumption exceeds maximum electric power capacity.")

        # Update consumption
        new_state = state.copy()
        new_state.consumption = load_consumption 
        new_state.consumption = min(new_state.consumption, self.max_electric_power)  # Cap consumption
        new_state.consumption = max(new_state.consumption, 0)  # Ensure consumption is non-negative

        # Apply natural decay or other dynamics if necessary
        # Placeholder for additional dynamics

        # Advance time
        new_state.current_time_step, new_state.hour = move_time_tick(
            state.current_time_step, state.hour
        )

        return new_state

    def predict(
        self,
        action: Union[ArrayLike, ConsumeAction],
        state: Optional[ConsumptionState] = None,
        **kwargs: Any
    ) -> ConsumptionState:
        """
        Predict the next consumption state based on the given action and current state.

        This method simulates the consumption without mutating the original state.

        Parameters
        ----------
        action : Union[ArrayLike, ConsumeAction]
            The action to perform. Must be either a NumPy array with a single value
            or a `ConsumeAction` instance.
        state : Optional[ConsumptionState], default=None
            The current state of consumption. If `None`, a default state must be provided.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        ConsumptionState
            The predicted consumption state after performing the action.

        Raises
        ------
        ValueError
            If the `action` is invalid or `state` is not provided.
        TypeError
            If the `action` is not of the expected type.
        """
        if state is None:
            raise ValueError("Consumption state must be provided for prediction.")

        # Validate and process action
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError("Action array must contain exactly one value.")
            action_value = ConsumeAction.from_numpy(action)
        elif isinstance(action, ConsumeAction):
            action_value = action
        else:
            raise TypeError("Action must be a numpy array or ConsumeAction instance.")

        consumption_adjustment = action_value.consume

        # Determine the current day index
        if state.current_time_step % self.num_samples_per_day == 0:
            num_days = len(self.load_data) // self.num_samples_per_day
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
        load_consumption = self.load_data.iloc[idx]

        # Validate load data
        if not isinstance(load_consumption, (float, int)):
            raise TypeError("Load data must be a numeric value.")
        if load_consumption < 0:
            load_consumption = 0
            # Optionally, log a warning about negative load data being set to zero
        if load_consumption > self.max_electric_power:
            raise ValueError("Load consumption exceeds maximum electric power capacity.")

        # Predict consumption
        predicted_state = state.copy()
        predicted_state.consumption = load_consumption + consumption_adjustment
        predicted_state.consumption = min(predicted_state.consumption, self.max_electric_power)  # Cap consumption
        predicted_state.consumption = max(predicted_state.consumption, 0)  # Ensure consumption is non-negative

        # Apply natural decay or other dynamics if necessary
        # Placeholder for additional dynamics

        # Advance time
        predicted_state.current_time_step, predicted_state.hour = move_time_tick(
            state.current_time_step, state.hour
        )

        return predicted_state

    def get_current_consumption_capability(self) -> float:
        """
        Get the current consumption capability of the load.

        Returns
        -------
        float
            The current consumption capability in [kW].
        """
        return self.max_electric_power

    def predict_consumption_capability(self, state: ConsumptionState) -> float:
        """
        Predict the future consumption capability based on the current state.

        Parameters
        ----------
        state : ConsumptionState
            The current state of consumption.

        Returns
        -------
        float
            The predicted consumption capability in [kW].

        Notes
        -----
        This method currently returns the maximum electric power as the predicted capability.
        Future implementations can include more sophisticated prediction logic.
        """
        # Placeholder for more sophisticated prediction logic
        # For simplicity, return the maximum electric power
        return self.max_electric_power
