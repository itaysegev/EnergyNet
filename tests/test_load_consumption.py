import unittest
import os
import pandas as pd
from energy_net.dynamics.consumption_dynamics.consumption_dynamics import GeneralLoad
from energy_net.data.data import TimeSeriesData, DATASETS_DIRECTORY, DATA_DIRECTORY

class TestConsumption(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a path for the test file
        cls.test_file = os.path.join(DATASETS_DIRECTORY, 'CAISO_net-load_2021.xlsx')
    
    def testConsumption(self):
        consumer = GeneralLoad()
        time_step = 59
        expected_value = 19670.8083333333
        consumption = consumer.do_data_driven(time_step=time_step)
        self.assertAlmostEqual(expected_value, consumption)

if __name__ == '__main__':
    unittest.main()