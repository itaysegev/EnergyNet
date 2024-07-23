from scipy.integrate import quad
from scipy.misc import derivative
from typing import Callable, Any, TypedDict
import numpy as np
import matplotlib.pyplot as plt

from ..model.state import State

AggFunc = Callable[[list[dict[str, Any]]], dict[str, Any]]


def agg_func_sum(element_arr:list[dict[str, Any]])-> dict[str, Any]:
    sum_dict = {}
    for element in element_arr:
        for entry in element:
            if entry in sum_dict.keys():
                sum_dict[entry] += element[entry]
            else:
                sum_dict[entry] = element[entry]
    return sum_dict


def condition(state:State):
    pass


def get_predicted_state(cur_state:State, horizon:float)->State:
    state = State({'time':cur_state['time']+horizon})
    return state


def get_value_by_type(dict, wanted_type):
    print(dict)
    print(wanted_type)
    for value in dict.values():
        if type(value) is wanted_type:
            return value
    
    return None

def unit_conversion(dest_units: str, x: float, T: tuple[float, float]) -> float:
    """
    Function for unit conversion. Calculate energy by integrating the power function
    over the specified time interval. Calculate energy by derivating the energy function.

    Parameters:
        dest_units : Indicates the direction of the conversion.
        x : function
            May be the power function as a function of time or the energy function as function of time.
        T : tuple
            A tuple representing the time interval (start, end).

    Returns:
        float
            The calculated energy or power.
    """
    if dest_units == 'W':
        # Differentiate the energy function over the time interval
        y = derivative(x, T, dx=1e-6, n=1)
    elif dest_units == 'J':
        # Integrate the power function over the time interval
        y, _ = quad(x, T[0], T[1])
    return y

def move_time_tick(cur_time, cur_hour):
    new_time = cur_time + 1
    if new_time % 2 == 0:
        cur_hour += 1
    if cur_hour == 24:
        cur_hour = 0
    return new_time, cur_hour 
    
def plot_data(data, title):
    """
    Plots the given data against the step number.

    Args:
        data (list): A list containing the data to be plotted.
        title (str): The title for the plot.
    """
    # Create a list of steps
    steps = list(range(len(data)))

    # Create a new figure
    plt.figure(figsize=(8, 6))

    # Plot the data
    plt.plot(steps, data)

    # Add title and labels
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel(title)

    # Show the plot
    plt.show()


def plot(train_rewards, eval_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(train_rewards, label='Training Rewards')
    plt.plot(eval_rewards, label='Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training and Evaluation Rewards')
    plt.legend()
    plt.show()
    
    
def hourly_pricing(hour):
    if hour < 6:
        return 12
    elif hour < 10:
        return 14
    elif hour < 20:
        return 20
    else:
        return 14

