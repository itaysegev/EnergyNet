from abc import abstractmethod

from .params import DynamicsParams
from ..config import DEFAULT_PRODUCTION
from ..model.state import State
from ..model.action import EnergyAction


class EnergyDynamics():

    def __init__(self, dynamics_params:DynamicsParams = None):
        """
        Constructor for the NetworkEntity class.

        Parameters:
        name (str): The name of the network entity.
        """
        self.dynamics_params = dynamics_params
    @abstractmethod
    def do(self, action: EnergyAction, state:State = None, params = None):
        pass

    @abstractmethod
    def predict(self, action: EnergyAction, state:State = None, params = None):
        pass

    @abstractmethod
    def reset(self):
        pass


class ProductionDynamics(EnergyDynamics):

   def __init__(self, dynamics_params:DynamicsParams = None):
    super().__init__(dynamics_params)

    @abstractmethod
    def do(self, action: EnergyAction, state: State = None, params = None):
        pass

    @abstractmethod
    def predict(self, action: EnergyAction, state: State = None, params = None):
        pass

    @abstractmethod
    def get_current_production_capability(self):
        pass

    @abstractmethod
    def predict_production_capability(self, state: State):
        pass



class ConsumptionDynamics(EnergyDynamics):

    def __init__(self, dynamics_params: DynamicsParams = None):
        super().__init__(dynamics_params)

    @abstractmethod
    def do(self, action: EnergyAction, state: State= None, params= None):
        pass

    @abstractmethod
    def predict(self, action: EnergyAction, state: State= None, params= None):
        pass

    @abstractmethod
    def get_current_consumption_capability(self):
        pass

    @abstractmethod
    def predict_consumption_capability(self, state:State):
        pass

class StorageDynamics(EnergyDynamics):

    def __init__(self, dynamics_params: DynamicsParams = None):
        super().__init__(dynamics_params)

    @abstractmethod
    def do(self, action: EnergyAction, state: State=None, params= None):
        pass

    @abstractmethod
    def predict(self, action: EnergyAction, state: State=None, params= None):
        pass


    @abstractmethod
    def get_current_discharge_capability(self):
        pass

    @abstractmethod
    def predict_discharge_capability(self, state:State):
        pass

    @abstractmethod
    def get_current_charge_capability(self):
        pass

    @abstractmethod
    def predict_charge_capability(self, state:State):
        pass

class TransmissionDynamics(EnergyDynamics):
    pass


class ComplexDynamics(EnergyDynamics):

    def __init__(self, dynamics_params: DynamicsParams = None):
        super().__init__(dynamics_params)

    @abstractmethod
    def do(self, action: EnergyAction, state: State, sub_entities_dynamics:list[EnergyDynamics]):
        pass

    @abstractmethod
    def predict(self, action: EnergyAction, state: State, sub_entities_dynamics:list[EnergyDynamics]):
        pass
