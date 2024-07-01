# Write unittest for the following classes: EpisodeTracker, Environment, EnergyNetEnv, PrivateProducer

import unittest
from energy_net.env.base import EpisodeTracker, Environment
# from energy_net.env.EnergyNetEnv import EnergyNetEnv

class TestEpisodeTracker(unittest.TestCase):
    def setUp(self):
        self.episode_tracker = EpisodeTracker(simulation_end_time_step=100, simulation_start_time_step=0)

    def test_initialization(self):
        self.assertEqual(self.episode_tracker.simulation_start_time_step, 0)
        self.assertEqual(self.episode_tracker.simulation_end_time_step, 100)
        self.assertEqual(self.episode_tracker.simulation_time_steps, 101)
        self.assertEqual(self.episode_tracker.episode, -1)


    def test_next_episode(self):
        self.episode_tracker.next_episode(50, False, False, 0)
        self.assertEqual(self.episode_tracker.episode_time_steps, 50)
        self.assertEqual(self.episode_tracker.episode_start_time_step, 0)
        self.assertEqual(self.episode_tracker.episode_end_time_step, 49)
        self.assertEqual(self.episode_tracker.episode_time_steps, 50)
        self.assertEqual(self.episode_tracker.episode, 0)

        self.episode_tracker.next_episode(50, False, False, 0) 
        self.assertEqual(self.episode_tracker.episode_time_steps, 50)
        self.assertEqual(self.episode_tracker.episode_start_time_step, 50)
        self.assertEqual(self.episode_tracker.episode_end_time_step, 99)
        self.assertEqual(self.episode_tracker.episode_time_steps, 50)
        self.assertEqual(self.episode_tracker.episode, 1)


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        episode_tracker = EpisodeTracker(simulation_end_time_step=100, simulation_start_time_step=0)
        self.environment = Environment(seconds_per_time_step=60*30, episode_tracker=episode_tracker, random_seed=0)

    def test_initialization(self):
        self.assertEqual(self.environment.DEFAULT_RANDOM_SEED_RANGE, (0, 100_000_000))
        self.assertEqual(self.environment.seconds_per_time_step, 60*30)
        self.assertEqual(self.environment.random_seed, 0)
        

    def test_uid(self):
        self.assertEqual(type(self.environment.uid), str)

    def test_episode_tracker(self):
        self.assertEqual(self.environment.episode_tracker.simulation_end_time_step, 100)

    def test_time_step(self):
        self.environment.reset_time_step()
        self.assertEqual(self.environment.time_step, 0)

        self.environment.next_time_step()
        self.assertEqual(self.environment.time_step, 1)

        self.environment.reset_time_step()
        self.assertEqual(self.environment.time_step, 0)

 

class TestEnergyNetEnv(unittest.TestCase):
    def setUp(self):
        self.energy_net_env = EnergyNetEnv()

    def test_initialization(self):
        self.assertEqual(self.energy_net_env.metadata, {"name": "energy_net_env_v0"})
        self.assertEqual(self.energy_net_env.episode_tracker.simulation_start_time_step, None)
        self.assertEqual(self.energy_net_env.episode_tracker.simulation_end_time_step, None)
        self.assertEqual(self.energy_net_env.episode_tracker.episode_time_steps, None)
        self.assertEqual(self.energy_net_env.episode_tracker.seconds_per_time_step, None)
        self.assertEqual(self.energy_net_env.episode_tracker.random_seed, None)
        self.assertEqual(self.energy_net_env.network_lst, [])
        self.assertEqual(self.energy_net_env.num_entities, 0)
        self.assertEqual(self.energy_net_env.episode_time_steps, None)
        self.assertEqual(self.energy_net_env.__state, None)
        self.assertEqual(self.energy_net_env.__np_random, None)
        self.assertEqual(self.energy_net_env.entities, {})
        self.assertEqual(self.energy_net_env.agents, [])
        self.assertEqual(self.energy_net_env.agents_name_to_network, {})
        self.assertEqual(self.energy_net_env.__observation_space, None)
        self.assertEqual(self.energy_net_env.__action_space, None)

    def test_uid(self):
        self.assertEqual(type(self.energy_net_env.uid), str)

    def test_random_seed(self):
        self.assertEqual(self.energy_net_env.random_seed, None)

    def test_episode_tracker(self):
        self.assertEqual(self.energy_net_env.episode_tracker, None)

    def test_time_step(self):
        self.assertEqual(self.energy_net_env.time_step, None)

    def test_seconds_per_time_step(self):
        self.assertEqual(self.energy_net_env.seconds_per_time_step, None)




if __name__ == '__main__':
    unittest.main()