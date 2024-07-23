from typing import Callable, Any, TypedDict
import numpy as np

AmountPricePair = tuple[float, float]
PriceList = list[AmountPricePair]
ProductionPredFn = Callable[[Any, ...], PriceList]
ProductionFn = Callable[[Any, ...], AmountPricePair]

Bid = [float, float]

class Bounds:
    def __init__(self, low: Any, high: Any, dtype: Any, shape: Any):
        self.low = low
        self.high = high
        self.dtype = dtype
        self.shape = shape
        
    def remove_first_dim(self):
        """
        Remove the first dimension from both `low` and `high`, and update `shape`.
        """
        if isinstance(self.low, np.ndarray) and isinstance(self.high, np.ndarray):
            self.low = self.low[1:]
            self.high = self.high[1:]
        elif isinstance(self.low, list) and isinstance(self.high, list):
            self.low = self.low[1:]
            self.high = self.high[1:]
        else:
            raise TypeError("Unsupported type for `low` and `high`. Must be list or np.ndarray.")
        
        if isinstance(self.shape, tuple):
            self.shape = self.shape[1:]
        else:
            raise TypeError("Unsupported type for `shape`. Must be tuple.")

      
