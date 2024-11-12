# main.py

from energy_net.market_entity import ControlledProducer, MarketStorage
import matplotlib.pyplot as plt
import numpy as np


def test_market_with_production():
    # Define time intervals (48 half-hour intervals for a day)
    intervals = range(48)

    # Predicted demand and prices for each interval
    predicted_demand = [150 + 50 * np.cos((i + 5) * np.pi / 24) for i in intervals]
    predicted_prices = [50 + 20 * np.cos((i + 5) * np.pi / 24) for i in intervals]

    # Initialize market players with access to predicted demand and prices
    producer = ControlledProducer(predicted_demand, predicted_prices, production_capacity=200)
    # consumer = MarketConsumer(predicted_demand, predicted_prices, consumption_demand=80)
    storage = MarketStorage(predicted_demand, predicted_prices, storage_capacity=100, initial_storage=0, charge_rate=100,
                            discharge_rate=100)

    # History lists for plotting
    producer_production_history = []
    storage_discharge_history = []
    storage_charge_history = []
    total_demand_history = []
    total_production_history = []
    time_stamps = []

    # For debugging purposes
    storage_levels = []

    # Main simulation loop
    for i in intervals:
        # Predicted demand at this timestamp
        predicted_demand_i = predicted_demand[i]

        # Storage's action
        storage_action = storage.decide_action(i)

        # Adjusted demand based on storage action
        if storage_action > 0:
            # Storage is discharging (producing energy)
            adjusted_demand = max(0, predicted_demand_i - storage_action)
            storage_discharge = storage_action
            storage_charge = 0
        elif storage_action < 0:
            # Storage is charging (consuming energy)
            adjusted_demand = predicted_demand_i - storage_action  # Subtracting negative number
            storage_charge = -storage_action  # Store as positive value
            storage_discharge = 0
        else:
            adjusted_demand = predicted_demand_i
            storage_discharge = 0
            storage_charge = 0

        # Producer's production
        producer_production = producer.decide_action(i)
        adjusted_demand = max(0, predicted_demand_i - storage_action)
        producer_production_history.append(adjusted_demand)

        # Record storage actions
        storage_discharge_history.append(storage_discharge)
        storage_charge_history.append(storage_charge)

        # Total production
        total_production = producer_production + storage_discharge
        total_production_history.append(total_production)

        # Total demand (for reference)
        total_demand = adjusted_demand + storage_charge  # Include storage charge in demand
        total_demand_history.append(total_demand)

        # Record storage level for debugging
        storage_levels.append(storage.current_storage)

        time_stamps.append(i)

    # Debugging: Print storage actions and levels
    print("Storage Discharge History:", storage_discharge_history)
    print("Storage Charge History:", storage_charge_history)
    print("Storage Levels Over Time:", storage_levels)

    # Plotting the bar graph
    bar_width = 0.5
    index = np.arange(len(time_stamps))

    plt.figure(figsize=(15, 7))

    # Producer's production in light blue
    plt.bar(index, producer_production_history, bar_width, color='lightblue', label='Producer Production')

    # Storage discharge (green) stacked on top of producer's production up to total demand
    bottom_discharge = producer_production_history
    plt.bar(index, storage_discharge_history, bar_width, bottom=producer_production_history, color='green',
            label='Storage Discharge')

    # Storage charge (dark blue) stacked on top to reflect additional consumption
    # For storage charging, increase the total demand bar
    total_consumption = np.array(producer_production_history) + np.array(storage_discharge_history) + np.array(
        storage_charge_history)
    plt.bar(index, storage_charge_history, bar_width, bottom=total_production_history, color='darkblue',
            label='Storage Charge')

    # Draw a line for predicted demand
    plt.plot(index, predicted_demand, color='red', linestyle='--', label='Predicted Demand')

    plt.xlabel('Time Interval (Half-Hours)')
    plt.ylabel('Energy (Units)')
    plt.title('Energy Production and Storage Actions Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Run the test function
if __name__ == "__main__":
    test_market_with_production()


