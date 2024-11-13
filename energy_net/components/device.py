"""
This code is based on https://github.com/intelligent-environments-lab/CityLearn/blob/master/citylearn/energy_model.py
"""

from abc import abstractmethod
from typing import Any, Dict

import numpy as np

from .params import DeviceParams
from energy_net.grid_entity import ElementaryGridEntity
from ..config import DEFAULT_LIFETIME_CONSTANT
from ..model.state import State
from energy_net.defs import Bounds

# Suppress specific numpy warnings globally
# Consider handling warnings locally where they occur instead
np.seterr(divide='ignore', invalid='ignore')


<<<<<<< HEAD:energy_net/components/device.py
class Device(ElementaryGridEntity):
    """Base device class.
=======
class Device(ElementaryNetworkEntity):
    """
    Represents a generic device within the smart grid network.

    This abstract base class defines the common attributes and behaviors
    for all devices in the network, including energy dynamics and lifetime management.

    Parameters
    ----------
    device_params : DeviceParams
        A dictionary containing parameters to initialize the device.
        Expected keys:
            - "name" (str): The name of the device.
            - "energy_dynamics" (Any): The energy dynamics configuration.
            - "lifetime_constant" (float, optional): The lifetime constant of the device. Defaults to DEFAULT_LIFETIME_CONSTANT.

    Attributes
    ----------
    name : str
        The name of the device.
    energy_dynamics : Any
        The energy dynamics configuration of the device.
    lifetime_constant : float
        The lifetime constant representing the device's technical efficiency.
    action_type : Any
        The type of action the device can perform. To be defined in subclasses.
>>>>>>> 56e876b696ab19dfab192159662e835d3d96a84f:energy_net/devices/device.py
    """

    def __init__(self, device_params: DeviceParams, init_state: State):
        """
        Initialize the Device with given parameters and initial state.

        Parameters
        ----------
        device_params : DeviceParams
            A dictionary containing initialization parameters.
        init_state : State
            The initial state of the device.
        """
        name = device_params.get("name")
        energy_dynamics = device_params.get("energy_dynamics")
        super().__init__(name, energy_dynamics, init_state=init_state)

        self.lifetime_constant = device_params.get(
            "lifetime_constant", DEFAULT_LIFETIME_CONSTANT
        )
        self.action_type = None  # To be defined in subclasses

    @property
    def lifetime_constant(self) -> float:
        """
        float: The lifetime constant representing the device's technical efficiency.
        """
        return self.__lifetime_constant

    @lifetime_constant.setter
    def lifetime_constant(self, lifetime_constant: float) -> None:
        """
        Set the lifetime constant of the device.

        Parameters
        ----------
        lifetime_constant : float
            The new lifetime constant value.

        Raises
        ------
        ValueError
            If `lifetime_constant` is not a positive float.
        """
        if not isinstance(lifetime_constant, (float, int)):
            raise TypeError("lifetime_constant must be a float.")
        if lifetime_constant < 0:
            raise ValueError("lifetime_constant must be a positive value.")
        self.__lifetime_constant = float(lifetime_constant)

    def dynamics_parameters(self) -> Dict[str, float]:
        """
        Retrieve the dynamics parameters of the device.

        Returns
        -------
        Dict[str, float]
            A dictionary containing dynamics parameters.
        """
        return {"lifetime_constant": self.__lifetime_constant}

    def reset(self) -> None:
        """
        Reset the device to its initial state.

        This method resets the device's state by calling the parent class's reset method.
        """
        super().reset()

    @abstractmethod
    def get_observation_space(self) -> Bounds:
        """
        Define the observation space for the device.

        Must be implemented by subclasses.

        Returns
        -------
        Bounds
            The bounds of the observation space.
        """
        pass

    @abstractmethod
    def get_action_space(self) -> Bounds:
        """
        Define the action space for the device.

        Must be implemented by subclasses.

        Returns
        -------
        Bounds
            The bounds of the action space.
        """
        pass
