from abc import abstractmethod
from typing import Any, List

from .params import DynamicsParams
from ..config import DEFAULT_PRODUCTION
from ..model.state import State
from ..model.action import EnergyAction
from ..data.data import TimeSeriesData

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
    
    
    
class DataDrivenDynamics(EnergyDynamics):
    def __init__(self, dynamics_params: DynamicsParams = None, start_time_step: int = None, end_time_step: int = None):
        super().__init__(dynamics_params)
        
        # Assert the data file path is provided in dynamics_params
        if not dynamics_params or not dynamics_params.data_file:
            raise ValueError("Data file path must be specified in dynamics_params")
        
        # Initialize TimeSeriesData with the given parameters
        self.time_series_data = TimeSeriesData(dynamics_params.data_file, start_time_step, end_time_step)
    
    @abstractmethod
    def do(self, action: EnergyAction, state: State = None, params: Any = None):
        # Implement the logic for the 'do' method
        pass
    @abstractmethod
    def predict(self, action: EnergyAction, state: State = None, params: Any = None):
        # Implement the logic for the 'predict' method
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
