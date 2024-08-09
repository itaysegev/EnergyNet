

def losses_simulation_reward_function(state, action, new_state):
    alpha = 0.01
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    price = grid_electricity
    price = price + alpha * price * price
    return -1 * price * price

def simulation_reward_function(state, action, new_state):
    
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    
    if grid_electricity < 0 or new_state.get_soc() - state.get_soc() != action.item():
        return -1_000
    price = grid_electricity
    return -1 * price * grid_electricity 



def fixed_prices_reward_function(state, action, new_state):
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    price = 630
    if grid_electricity < 0:
        return -10_000
    return -1 * price * grid_electricity


def hourly_prices_reward_function(state, action, new_state):
    hour = state.get_hour()
    
    if 17 <= hour <= 23:
        price = 1660
    else:
        price = 490
        
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    if grid_electricity < 0:
        return -100_000
    return -1 * price * grid_electricity


def losses_simulation_reward_function_with_soc(state, action, new_state):
    alpha = 0.01
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    price = grid_electricity
    price = price + alpha * price * price
    
    # Check SOC in the last hour
    if state.get_hour() == 23 and state.get_soc() != 0:
        return -10_000  # Very low reward if SOC is not zero

    return -1 * price * price



def simulation_reward_function_with_soc(state, action, new_state):
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    
    if grid_electricity < 0:
        return -1_000
    
    # Check SOC in the last hour
    if new_state.get_hour() == 0 and new_state.get_soc() > 0.01:
        return -10_000  # Very low reward if SOC is not zero

    price = grid_electricity
    return -1 * price * grid_electricity



def fixed_prices_reward_function_with_soc(state, action, new_state):
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    price = 630
    
    if grid_electricity < 0:
        return -10_000
    
    # Check SOC in the last hour
    if state.get_hour() == 23 and state.get_soc() != 0:
        return -10_000  # Very low reward if SOC is not zero

    return -1 * price * grid_electricity


def hourly_prices_reward_function_with_soc(state, action, new_state):
    hour = state.get_hour()
    
    if 17 <= hour <= 23:
        price = 1660
    else:
        price = 490
        
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    
    if grid_electricity < 0:
        return -100_000
    
    # Check SOC in the last hour
    if state.get_hour() == 23 and state.get_soc() != 0:
        return -10_000  # Very low reward if SOC is not zero

    return -1 * price * grid_electricity


    