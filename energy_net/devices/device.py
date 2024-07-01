'''This code is based on https://github.com/intelligent-environments-lab/CityLearn/blob/master/citylearn/energy_model.py'''
from abc import abstractmethod
import numpy as np



from .params import DeviceParams
from ..entities.network_entity import ElementaryNetworkEntity
from ..config import DEFAULT_LIFETIME_CONSTANT
from ..model.state import State
from energy_net.defs import Bounds

np.seterr(divide='ignore', invalid='ignore')


class Device(ElementaryNetworkEntity):
    """Base device class.
    """

    def __init__(self, device_params:DeviceParams, init_state:State):
        super().__init__(device_params["name"], device_params["energy_dynamics"], init_state = init_state)
        self.__lifetime_constant = device_params["lifetime_constant"] if "lifetime_constant" in device_params else DEFAULT_LIFETIME_CONSTANT
        

    @property
    def lifetime_constant(self) -> float:
        """Technical efficiency."""
        return self.__lifetime_constant

    @lifetime_constant.setter
    def lifetime_constant(self, life_time_constant: float):
        self.__lifetime_constant = life_time_constant

    
    def dynamics_parameters(self):
        return {'lifetime_constant': self.__lifetime_constant}
    
    
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_observation_space(self) -> Bounds:
        pass

    @abstractmethod
    def get_action_space(self) -> Bounds:
        pass
    
    
        

    

        

    
    
        



        

