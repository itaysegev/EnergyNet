import gymnasium as gym
import os
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env

from utils import create_gym_env, train_and_save_models, collect_observations, save_observations_to_excel, build_pcsunit
from rewards import simulation_reward_function, losses_simulation_reward_function, fixed_prices_reward_function, hourly_prices_reward_function, losses_simulation_reward_function_with_soc, simulation_reward_function_with_soc, fixed_prices_reward_function_with_soc, hourly_prices_reward_function_with_soc

def run_test_case(efficacy, reward_function, name):
    
    models_path = "Models"
    output_path = "Excels"
    
    # # Create train environment
    # train_env = create_gym_env("train_network", build_pcsunit('train_data.xlsx', efficiency=efficacy), reward_function)

    # Create test environment
    test_env = create_gym_env("test_network", build_pcsunit('test_data.xlsx', efficiency=efficacy), reward_function)

    # Train and save models
    model_save_path = os.path.join(models_path, name)
    # train_and_save_models(train_env, model_save_path)
    
    # Load the trained models
    model_ppo = PPO.load(os.path.join(model_save_path, "ppo_model"))
    model_sac = SAC.load(os.path.join(model_save_path, "sac_model"))
    model_td3 = TD3.load(os.path.join(model_save_path, "td3_model"))

    # Collect observations
    ppo_observations = collect_observations(test_env, model_ppo)
    sac_observations = collect_observations(test_env, model_sac)
    td3_observations = collect_observations(test_env, model_td3)
    
    
    # Save observations to Excel
    output_path = os.path.join(output_path, name)
    save_observations_to_excel(ppo_observations, sac_observations, td3_observations, output_path)
    
    
    
def main():
    
    ################################ Original Test cases ################################
    
    # First test case
    name = "Original_no_first"
    run_test_case(efficacy=1, reward_function=simulation_reward_function, name=name)
    
    # Second test case
    name = "Original_no_second"
    run_test_case(efficacy=0.9, reward_function=simulation_reward_function, name=name)
    
    # Third test case
    name = "Original_no_third"
    run_test_case(efficacy=0.9, reward_function=losses_simulation_reward_function, name=name)
    
    # ################################################################################################
    
    # ################################ Modified Test cases ################################
    
    # First test case
    name = "Original_yes_first"
    run_test_case(efficacy=1, reward_function=simulation_reward_function_with_soc, name=name)
    
    # Second test case
    name = "Original_yes_second"
    run_test_case(efficacy=0.9, reward_function=simulation_reward_function_with_soc, name=name)
    
    # Third test case
    name = "Original_yes_third"
    run_test_case(efficacy=0.9, reward_function=losses_simulation_reward_function_with_soc, name=name)
    
    
    
    ################################################################################################
    ################################### Fixed Prices Test cases ###################################
    
    # First test case
    name = "Fixed_no_first"
    run_test_case(efficacy=1, reward_function=fixed_prices_reward_function, name=name)
    
    # Second test case
    name = "Fixed_no_second"
    run_test_case(efficacy=0.9, reward_function=fixed_prices_reward_function, name=name)
        
    ################################################################################################
    
    # First test case
    name = "Fixed_yes_first"
    run_test_case(efficacy=1, reward_function=fixed_prices_reward_function_with_soc, name=name)
    
    # Second test case
    name = "Fixed_yes_second"
    run_test_case(efficacy=0.9, reward_function=fixed_prices_reward_function_with_soc, name=name)
    

    
    ################################################################################################
    ################################### Hourly Prices Test cases ###################################
    
    # First test case
    name = "Hourly_no_first"
    run_test_case(efficacy=1, reward_function=hourly_prices_reward_function, name=name)
    
    # Second test case
    name = "Hourly_no_second"
    run_test_case(efficacy=0.9, reward_function=hourly_prices_reward_function, name=name)
     
    ################################################################################################
    
    # First test case
    name = "Hourly_yes_first"
    run_test_case(efficacy=1, reward_function=hourly_prices_reward_function_with_soc, name=name)
    
    # Second test case
    name = "Hourly_yes_second"
    run_test_case(efficacy=0.9, reward_function=hourly_prices_reward_function_with_soc, name=name)
       
    ################################################################################################
    
if __name__ == "__main__":
    main()