"""
Energy Dynamics Module for Smart Grid Simulation.

This module defines various energy dynamics classes that model the behavior of different
energy devices within the smart grid system. It utilizes abstract base classes to enforce
the implementation of essential methods in derived classes.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

from .params import DynamicsParams
from ..config import (
    DEFAULT_PRODUCTION,
    DEFAULT_LIFETIME_CONSTANT,
    DEFAULT_EFFICIENCY,
    DEFAULT_INIT_POWER,
    DEFAULT_SELF_CONSUMPTION,
)
from ..model.state import State
from ..model.action import EnergyAction
from ..data.data import TimeSeriesData


class EnergyDynamics(ABC):
    """
    Abstract Base Class for Energy Dynamics.

    This class defines the interface for energy dynamics, requiring the implementation
    of `do` and `predict` methods in derived classes.

    Attributes
    ----------
    dynamics_params : Optional[DynamicsParams]
        Parameters defining the dynamics behavior. Defaults to `None`.
    """

    def __init__(self, dynamics_params: Optional[DynamicsParams] = None):
        """
<<<<<<< HEAD
        Constructor for the GridEntity class.
=======
        Initialize the EnergyDynamics instance.
>>>>>>> 56e876b696ab19dfab192159662e835d3d96a84f

        Parameters
        ----------
        dynamics_params : Optional[DynamicsParams], default=None
            Parameters defining the dynamics behavior.
        """
        self.dynamics_params = dynamics_params

    @abstractmethod
    def do(
        self,
        action: EnergyAction,
        state: Optional[State] = None,
        params: Optional[Any] = None,
    ) -> None:
        """
        Execute an action to update the state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : Optional[State], default=None
            The current state before performing the action.
        params : Optional[Any], default=None
            Additional parameters for the action.
        """
        pass

    @abstractmethod
    def predict(
        self,
        action: EnergyAction,
        state: Optional[State] = None,
        params: Optional[Any] = None,
    ) -> State:
        """
        Predict the next state based on the given action and current state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : Optional[State], default=None
            The current state before performing the action.
        params : Optional[Any], default=None
            Additional parameters for the prediction.

        Returns
        -------
        State
            The predicted next state.
        """
        pass


class DataDrivenDynamics(EnergyDynamics):
    """
    Data-Driven Energy Dynamics.

    This class implements energy dynamics based on historical data provided through a
    time series data file.

    Attributes
    ----------
    time_series_data : TimeSeriesData
        The time series data used to drive the dynamics.
    """

    def __init__(
        self,
        dynamics_params: Optional[DynamicsParams] = None,
        start_time_step: Optional[int] = None,
        end_time_step: Optional[int] = None,
    ):
        """
        Initialize the DataDrivenDynamics instance.

        Parameters
        ----------
        dynamics_params : Optional[DynamicsParams], default=None
            Parameters defining the dynamics behavior, including the data file path.
        start_time_step : Optional[int], default=None
            The starting time step for the data.
        end_time_step : Optional[int], default=None
            The ending time step for the data.

        Raises
        ------
        ValueError
            If the data file path is not provided in dynamics_params.
        """
        super().__init__(dynamics_params)

        # Ensure that dynamics_params and data_file are provided
        if not dynamics_params or not hasattr(dynamics_params, 'data_file') or not dynamics_params.data_file:
            raise ValueError("Data file path must be specified in dynamics_params")

        # Initialize TimeSeriesData with the given parameters
        self.time_series_data = TimeSeriesData(
            dynamics_params.data_file,
            start_time_step=start_time_step,
            end_time_step=end_time_step,
        )

    @abstractmethod
    def do(
        self,
        action: EnergyAction,
        state: Optional[State] = None,
        params: Optional[Any] = None,
    ) -> None:
        """
        Execute an action to update the state based on data-driven dynamics.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : Optional[State], default=None
            The current state before performing the action.
        params : Optional[Any], default=None
            Additional parameters for the action.
        """
        pass

    @abstractmethod
    def predict(
        self,
        action: EnergyAction,
        state: Optional[State] = None,
        params: Optional[Any] = None,
    ) -> State:
        """
        Predict the next state based on the given action and current state using data-driven dynamics.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : Optional[State], default=None
            The current state before performing the action.
        params : Optional[Any], default=None
            Additional parameters for the prediction.

        Returns
        -------
        State
            The predicted next state.
        """
        pass


class ProductionDynamics(EnergyDynamics):
    """
    Production Energy Dynamics.

    This class defines the dynamics related to energy production devices.
    """

    def __init__(self, dynamics_params: Optional[DynamicsParams] = None):
        """
        Initialize the ProductionDynamics instance.

        Parameters
        ----------
        dynamics_params : Optional[DynamicsParams], default=None
            Parameters defining the production dynamics behavior.
        """
        super().__init__(dynamics_params)

    @abstractmethod
    def do(
        self,
        action: EnergyAction,
        state: Optional[State] = None,
        params: Optional[Any] = None,
    ) -> None:
        """
        Execute an action to update the production state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : Optional[State], default=None
            The current state before performing the action.
        params : Optional[Any], default=None
            Additional parameters for the action.
        """
        pass

    @abstractmethod
    def predict(
        self,
        action: EnergyAction,
        state: Optional[State] = None,
        params: Optional[Any] = None,
    ) -> State:
        """
        Predict the next production state based on the given action and current state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : Optional[State], default=None
            The current state before performing the action.
        params : Optional[Any], default=None
            Additional parameters for the prediction.

        Returns
        -------
        State
            The predicted next production state.
        """
        pass

    @abstractmethod
    def get_current_production_capability(self) -> float:
        """
        Get the current production capability of the device.

        Returns
        -------
        float
            The current production capability in [kW].
        """
        pass

    @abstractmethod
    def predict_production_capability(self, state: State) -> float:
        """
        Predict the production capability based on the given state.

        Parameters
        ----------
        state : State
            The current state of the device.

        Returns
        -------
        float
            The predicted production capability in [kW].
        """
        pass


class ConsumptionDynamics(EnergyDynamics):
    """
    Consumption Energy Dynamics.

    This class defines the dynamics related to energy consumption devices.
    """

    def __init__(self, dynamics_params: Optional[DynamicsParams] = None):
        """
        Initialize the ConsumptionDynamics instance.

        Parameters
        ----------
        dynamics_params : Optional[DynamicsParams], default=None
            Parameters defining the consumption dynamics behavior.
        """
        super().__init__(dynamics_params)

    @abstractmethod
    def do(
        self,
        action: EnergyAction,
        state: Optional[State] = None,
        params: Optional[Any] = None,
    ) -> None:
        """
        Execute an action to update the consumption state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : Optional[State], default=None
            The current state before performing the action.
        params : Optional[Any], default=None
            Additional parameters for the action.
        """
        pass

    @abstractmethod
    def predict(
        self,
        action: EnergyAction,
        state: Optional[State] = None,
        params: Optional[Any] = None,
    ) -> State:
        """
        Predict the next consumption state based on the given action and current state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : Optional[State], default=None
            The current state before performing the action.
        params : Optional[Any], default=None
            Additional parameters for the prediction.

        Returns
        -------
        State
            The predicted next consumption state.
        """
        pass

    @abstractmethod
    def get_current_consumption_capability(self) -> float:
        """
        Get the current consumption capability of the device.

        Returns
        -------
        float
            The current consumption capability in [kW].
        """
        pass

    @abstractmethod
    def predict_consumption_capability(self, state: State) -> float:
        """
        Predict the consumption capability based on the given state.

        Parameters
        ----------
        state : State
            The current state of the device.

        Returns
        -------
        float
            The predicted consumption capability in [kW].
        """
        pass


class StorageDynamics(EnergyDynamics):
    """
    Storage Energy Dynamics.

    This class defines the dynamics related to energy storage devices.
    """

    def __init__(self, dynamics_params: Optional[DynamicsParams] = None):
        """
        Initialize the StorageDynamics instance.

        Parameters
        ----------
        dynamics_params : Optional[DynamicsParams], default=None
            Parameters defining the storage dynamics behavior.
        """
        super().__init__(dynamics_params)

    @abstractmethod
    def do(
        self,
        action: EnergyAction,
        state: Optional[State] = None,
        params: Optional[Any] = None,
    ) -> None:
        """
        Execute an action to update the storage state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : Optional[State], default=None
            The current state before performing the action.
        params : Optional[Any], default=None
            Additional parameters for the action.
        """
        pass

    @abstractmethod
    def predict(
        self,
        action: EnergyAction,
        state: Optional[State] = None,
        params: Optional[Any] = None,
    ) -> State:
        """
        Predict the next storage state based on the given action and current state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : Optional[State], default=None
            The current state before performing the action.
        params : Optional[Any], default=None
            Additional parameters for the prediction.

        Returns
        -------
        State
            The predicted next storage state.
        """
        pass

    @abstractmethod
    def get_current_discharge_capability(self) -> float:
        """
        Get the current discharge capability of the storage device.

        Returns
        -------
        float
            The current discharge capability in [kW].
        """
        pass

    @abstractmethod
    def predict_discharge_capability(self, state: State) -> float:
        """
        Predict the discharge capability based on the given state.

        Parameters
        ----------
        state : State
            The current state of the storage device.

        Returns
        -------
        float
            The predicted discharge capability in [kW].
        """
        pass

    @abstractmethod
    def get_current_charge_capability(self) -> float:
        """
        Get the current charge capability of the storage device.

        Returns
        -------
        float
            The current charge capability in [kW].
        """
        pass

    @abstractmethod
    def predict_charge_capability(self, state: State) -> float:
        """
        Predict the charge capability based on the given state.

        Parameters
        ----------
        state : State
            The current state of the storage device.

        Returns
        -------
        float
            The predicted charge capability in [kW].
        """
        pass


class TransmissionDynamics(EnergyDynamics):
    """
    Transmission Energy Dynamics.

    This class defines the dynamics related to energy transmission within the grid.
    """

    def __init__(self, dynamics_params: Optional[DynamicsParams] = None):
        """
        Initialize the TransmissionDynamics instance.

        Parameters
        ----------
        dynamics_params : Optional[DynamicsParams], default=None
            Parameters defining the transmission dynamics behavior.
        """
        super().__init__(dynamics_params)

    @abstractmethod
    def do(
        self,
        action: EnergyAction,
        state: Optional[State] = None,
        params: Optional[Any] = None,
    ) -> None:
        """
        Execute an action to update the transmission state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : Optional[State], default=None
            The current state before performing the action.
        params : Optional[Any], default=None
            Additional parameters for the action.
        """
        pass

    @abstractmethod
    def predict(
        self,
        action: EnergyAction,
        state: Optional[State] = None,
        params: Optional[Any] = None,
    ) -> State:
        """
        Predict the next transmission state based on the given action and current state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : Optional[State], default=None
            The current state before performing the action.
        params : Optional[Any], default=None
            Additional parameters for the prediction.

        Returns
        -------
        State
            The predicted next transmission state.
        """
        pass


class ComplexDynamics(EnergyDynamics):
    """
    Complex Energy Dynamics.

    This class defines complex energy dynamics involving multiple sub-entities.
    """

    def __init__(
        self,
        dynamics_params: Optional[DynamicsParams] = None,
    ):
        """
        Initialize the ComplexDynamics instance.

        Parameters
        ----------
        dynamics_params : Optional[DynamicsParams], default=None
            Parameters defining the complex dynamics behavior.
        """
        super().__init__(dynamics_params)

    @abstractmethod
    def do(
        self,
        action: EnergyAction,
        state: State,
        sub_entities_dynamics: List[EnergyDynamics],
    ) -> None:
        """
        Execute an action involving multiple sub-entities to update the complex state.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : State
            The current state before performing the action.
        sub_entities_dynamics : List[EnergyDynamics]
            A list of sub-entity dynamics involved in the complex action.
        """
        pass

    @abstractmethod
    def predict(
        self,
        action: EnergyAction,
        state: State,
        sub_entities_dynamics: List[EnergyDynamics],
    ) -> State:
        """
        Predict the next complex state based on the given action and current state involving multiple sub-entities.

        Parameters
        ----------
        action : EnergyAction
            The action to perform.
        state : State
            The current state before performing the action.
        sub_entities_dynamics : List[EnergyDynamics]
            A list of sub-entity dynamics involved in the complex prediction.

        Returns
        -------
        State
            The predicted next complex state.
        """
        pass
