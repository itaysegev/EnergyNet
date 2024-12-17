from typing import Callable, Dict
from energy_net.dynamics.iso.iso_base import ISOBase

class QuadraticPricingISO(ISOBase):
    """
    ISO implementation that uses a quadratic function to determine prices based on demand or other factors.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, c: float = 50.0):
        if not isinstance(a, (float, int)):
            raise TypeError(f"a must be a float or int, got {type(a).__name__}")
        if not isinstance(b, (float, int)):
            raise TypeError(f"b must be a float or int, got {type(b).__name__}")
        if not isinstance(c, (float, int)):
            raise TypeError(f"c must be a float or int, got {type(c).__name__}")

        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def reset(self) -> None:
        pass

    def get_pricing_function(self, observation: Dict) -> Callable[[float, float], float]:
        demand = observation.get('demand', 1.0)
        price_buy = self.a * (demand ** 2) + self.b * demand + self.c
        price_sell = price_buy * 0.85

        def pricing(buy: float, sell: float) -> float:
            return (buy * price_buy) - (sell * price_sell)

        return pricing
