from typing import TypedDict
import numpy as np


class EnergyAction():
    pass

class StorageAction(EnergyAction):
    def __init__(self, charge: float = 0):
        self.charge = charge
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        if arr.size != 1:
            raise ValueError("Input array must have a single element")
        return cls(charge=float(arr[0]))

class ProduceAction(EnergyAction):
    def __init__(self, production: float = 0):
        self.production = production
    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        if arr.size != 1:
            raise ValueError("Input array must have a single element")
        return cls(production=float(arr[0]))

class ConsumeAction(EnergyAction):
    def __init__(self, consumption: float = 0):
        self.consumption = consumption

    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        if arr.size != 1:
            raise ValueError("Input array must have a single element")
        return cls(consumption=float(arr[0]))

class TradeAction(EnergyAction):
    def __init__(self, amount: float = 0):
        self.amount = amount