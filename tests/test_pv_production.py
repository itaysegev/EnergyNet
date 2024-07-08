import unittest
import os
import pandas as pd
from energy_net.dynamics.production_dynamics.production_dynamics import PVDynamics
from energy_net.data.data import TimeSeriesData, DATASETS_DIRECTORY, DATA_DIRECTORY

class TestPVProduction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a path for the test file
        cls.test_file = os.path.join(DATASETS_DIRECTORY, 'CAISO_net-load_2021.xlsx')
    
    def testProduction(self):
        pv = PVDynamics()
        time_step = 2980
        expected_value = 5823.55758138021
        pv_production = pv.do_data_driven(time_step=time_step)
        self.assertAlmostEqual(expected_value, pv_production)

if __name__ == '__main__':
    unittest.main()