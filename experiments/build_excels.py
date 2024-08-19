import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3
from utils import create_gym_env, train_and_save_models, collect_observations, build_pcsunit

from rewards import simulation_reward_function, losses_simulation_reward_function, fixed_prices_reward_function, hourly_prices_reward_function, losses_simulation_reward_function_with_soc, simulation_reward_function_with_soc, fixed_prices_reward_function_with_soc, hourly_prices_reward_function_with_soc

days = 100
samples_per_day = 48

def save_observations_to_excel(ppo_observations_all_days, sac_observations_all_days, td3_observations_all_days, output_file_path):
    # Prepare to collect data for all days
    data_all_days = {
        'Day': [], 'Hour': [], 'Load': [], 'PV': [], 'PPO soc': [], 'SAC soc': [], 'TD3 soc': []
    }
    
    for day in range(days):
        ppo_observations = ppo_observations_all_days[day]
        sac_observations = sac_observations_all_days[day]
        td3_observations = td3_observations_all_days[day]
        
        # Append data for each day
        data_all_days['Day'].extend([day + 1] * samples_per_day)
        data_all_days['Hour'].extend([obs[1] for obs in ppo_observations])   # Assuming obs[1] is the Hour
        data_all_days['Load'].extend([obs[3] for obs in ppo_observations])   # Assuming obs[3] is the Load
        data_all_days['PV'].extend([obs[10] for obs in ppo_observations])    # Assuming obs[10] is the PV
        data_all_days['PPO soc'].extend([obs[4] for obs in ppo_observations]) # Assuming obs[4] is the soc
        data_all_days['SAC soc'].extend([obs[4] for obs in sac_observations]) # Assuming obs[4] is the soc
        data_all_days['TD3 soc'].extend([obs[4] for obs in td3_observations]) # Assuming obs[4] is the soc

    # Create a DataFrame for all days
    df_all_days = pd.DataFrame(data_all_days)
    
    # Ensure the output file path ends with '.xlsx'
    if not output_file_path.endswith('.xlsx'):
        output_file_path += '.xlsx'
        
    # Create directory if it doesn't exist
    directory = os.path.dirname(output_file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save to an Excel file
    df_all_days.to_excel(output_file_path, index=False)

    print(f"Results saved to {output_file_path}")


def build_table_test_case(efficacy, reward_function, name):

    models_path = "experiments/Models"
    output_path = "experiments/Excels"

    model_save_path = os.path.join(models_path, name)

    # Load your models
    model_ppo = PPO.load(os.path.join(model_save_path, "ppo_model"))
    model_sac = SAC.load(os.path.join(model_save_path, "sac_model"))
    model_td3 = TD3.load(os.path.join(model_save_path, "td3_model"))

   

    # Assuming each day has 48 samples
    days = 100
    samples_per_day = 48

    # Prepare to save the results for all days
    ppo_observations_all_days = []
    sac_observations_all_days = []
    td3_observations_all_days = []

    # Set the directory where the Excel files should be saved
    output_dir = 'energy_net/data/datasets'
    
    # Load your data
    data = pd.read_excel(os.path.join(output_dir,'test_data.xlsx'))

    for day in range(days):
        # Get the data for the current day
        day_data = data.iloc[day * samples_per_day: (day + 1) * samples_per_day]
        
        # Save the day's data to a temporary Excel file
        day_file_name = os.path.join(output_dir, f'day_{day+1}.xlsx')
        day_data.to_excel(day_file_name, index=False)
        
        # Create the test environment using the provided function
        test_env = create_gym_env("test_network", build_pcsunit(day_file_name, efficiency=efficacy), reward_function)
        
        # Collect observations for each model
        obs_ppo = collect_observations(test_env, model_ppo)
        obs_sac = collect_observations(test_env, model_sac)
        obs_td3 = collect_observations(test_env, model_td3)
        
        # Append the observations for this day
        ppo_observations_all_days.append(obs_ppo)
        sac_observations_all_days.append(obs_sac)
        td3_observations_all_days.append(obs_td3)

    

    output_path = os.path.join(output_path, name)

    # Save the results to Excel or CSV
    save_observations_to_excel(ppo_observations_all_days, sac_observations_all_days, td3_observations_all_days, output_path)
    
    
    
    
def main():

    # ################################ Modified Test cases ################################
    
    # First test case
    name = "Original_yes_first"
    build_table_test_case(efficacy=1, reward_function=simulation_reward_function_with_soc, name=name)
    
    # First test case
    name = "Original_no_first"
    build_table_test_case(efficacy=1, reward_function=simulation_reward_function, name=name)
    
    
    
    # Second test case
    name = "Original_yes_second"
    build_table_test_case(efficacy=0.9, reward_function=simulation_reward_function_with_soc, name=name)
    
    # Third test case
    name = "Original_yes_third"
    build_table_test_case(efficacy=0.9, reward_function=losses_simulation_reward_function_with_soc, name=name)
    
    
    
if __name__ == "__main__":
    main()