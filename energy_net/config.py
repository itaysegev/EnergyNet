# constants.py

from typing import Final
import numpy as np

#################### GENERAL CONSTANTS ####################
INF: Final = float('inf')
MIN_TIME_STEP: Final = 0
MAX_TIME_STEP: Final = 1_000
###########################################################

#################### EFFICIENCY CONSTANTS ####################
DEFAULT_EFFICIENCY: Final[float] = 1.0
MAX_EFFICIENCY: Final[float] = 1.0
MIN_EFFICIENCY: Final[float] = 0.0
############################################################

#################### CHARGE CONSTANTS #######################
NO_CHARGE: Final[float] = 0.0
MIN_CHARGE: Final[float] = 0.0 
############################################################

#################### CAPACITY CONSTANTS ####################
MIN_CAPACITY: Final[float] = 0.0
MAX_CAPACITY: Final[float] = INF
############################################################

#################### POWER CONSTANTS #######################
MIN_POWER: Final[float] = 0.0
MAX_ELECTRIC_POWER: Final[float] = INF
############################################################

#################### PRODUCTION CONSTANTS ###################
MIN_PRODUCTION: Final[float] = 0.0
MAX_PRODUCTION: Final[float] = INF
DEFAULT_PRODUCTION: Final[float] = INF
############################################################

#################### CONSUMPTION CONSTANTS ##################
NO_CONSUMPTION: Final[float] = 0.0
MAX_CONSUMPTION: Final[float] = INF
############################################################

#################### TIME CONSTANTS #########################
INITIAL_TIME: Final[int] = 0
MAX_TIME: Final[float] = INF
INITIAL_HOUR: Final[int] = 0
MAX_HOUR: Final[int] = 23
############################################################

#################### PRICE CONSTANTS ########################
MIN_PRICE: Final[float] = 0.0
MAX_PRICE: Final[float] = 100.0
############################################################

#################### EXPONENT CONSTANTS ####################
MAX_EXPONENT: Final[int] = 200
MIN_EXPONENT: Final[int] = -200
############################################################

#################### SELF-CONSUMPTION CONSTANTS ############
DEFAULT_SELF_CONSUMPTION: Final[float] = 0.0
DEFAULT_LIFETIME_CONSTANT: Final[float] = 0.0
############################################################

#################### INIT POWER CONSTANTS ##################
DEFAULT_INIT_POWER: Final[float] = 0.0
############################################################

#################### OTHER CONSTANTS #######################
PRED_CONST_DUMMY: Final[int] = 100  # Placeholder for prediction constant
############################################################