from typing import Callable, Any, TypedDict

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

      
