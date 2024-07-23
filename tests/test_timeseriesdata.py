import unittest
import os
import pandas as pd

from energy_net.data.data import TimeSeriesData, DATASETS_DIRECTORY, DATA_DIRECTORY

class TestTimeSeriesData(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a sample CSV file for testing
        cls.test_file = os.path.join(DATASETS_DIRECTORY, 'test_data.csv')
        data = {
            'time': range(100),
            'value': range(100, 200),
            'category': ['A'] * 50 + ['B'] * 50
        }
        cls.df = pd.DataFrame(data)
        os.makedirs(DATASETS_DIRECTORY, exist_ok=True)
        cls.df.to_csv(cls.test_file, index=False)
    
    @classmethod
    def tearDownClass(cls):
        # Remove the sample CSV file after testing
        os.remove(cls.test_file)
    
    def test_initialization(self):
        ts_data = TimeSeriesData('test_data.csv')
        self.assertEqual(len(ts_data.data), 100)
        self.assertListEqual(list(ts_data.data.columns), ['time', 'value', 'category'])
    
    def test_initialization_with_slicing(self):
        ts_data = TimeSeriesData('test_data.csv', start_time_step=10, end_time_step=20)
        self.assertEqual(len(ts_data.data), 10)
        self.assertListEqual(list(ts_data.data['time']), list(range(10, 20)))
    
    def test_getattr(self):
        ts_data = TimeSeriesData('test_data.csv')
        self.assertListEqual(list(ts_data.time), list(range(100)))
        self.assertListEqual(list(ts_data.value), list(range(100, 200)))
        with self.assertRaises(AttributeError):
            ts_data.non_existing_column
    
    def test_setattr(self):
        ts_data = TimeSeriesData('test_data.csv')
        ts_data.new_column = [0] * 100
        self.assertListEqual(list(ts_data.new_column), [0] * 100)
    
    def test_summary(self):
        ts_data = TimeSeriesData('test_data.csv')
        summary = ts_data.summary()
        self.assertIn('value', summary.columns)
        self.assertEqual(summary.loc['mean', 'value'], 149.5)
    
    def test_get_column(self):
        ts_data = TimeSeriesData('test_data.csv')
        column_data = ts_data.get_column('value', start_time_step=10, end_time_step=20)
        self.assertListEqual(list(column_data), list(range(110, 120)))
    
    def test_add_column(self):
        ts_data = TimeSeriesData('test_data.csv')
        new_data = pd.Series([1] * 100)
        ts_data.add_column('new_col', new_data)
        self.assertListEqual(list(ts_data.new_col), [1] * 100)
    
    def test_save(self):
        ts_data = TimeSeriesData('test_data.csv')
        save_file = os.path.join(DATASETS_DIRECTORY, 'saved_data.csv')
        ts_data.save('saved_data.csv')
        self.assertTrue(os.path.exists(save_file))
        saved_data = pd.read_csv(save_file)
        self.assertTrue(saved_data.equals(ts_data.data))
        os.remove(save_file)

class TestTimeSeriesDataRealFile(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a path for the test file
        cls.test_file = os.path.join(DATASETS_DIRECTORY, 'CAISO_net-load_2021.xlsx')
    
    def test_initialization(self):
        ts_data = TimeSeriesData('CAISO_net-load_2021.xlsx')
        self.assertEqual(len(ts_data.data), len(pd.read_excel(self.test_file)))
        self.assertListEqual(list(ts_data.data.columns), [
            'Date', 'Hour', 'Interval', 'Load', 'Solar', 'Wind', 'Net Load',
            'Renewables', 'Nuclear', 'Large Hydro', 'Imports', 'Generation',
            'Thermal', 'Load Less (Generation+Imports)', 'Unnamed: 14'
        ])
    
    def test_initialization_with_slicing(self):
        ts_data = TimeSeriesData('CAISO_net-load_2021.xlsx', start_time_step=10, end_time_step=20)
        self.assertEqual(len(ts_data.data), 10)
        self.assertListEqual(list(ts_data.data['Interval']), [11,12] + list(range(1, 9)))
        
    
    def test_getattr(self):
        ts_data = TimeSeriesData('CAISO_net-load_2021.xlsx')
        self.assertListEqual(list(ts_data.Load), list(pd.read_excel(self.test_file)['Load']))
        self.assertListEqual(list(ts_data.Solar), list(pd.read_excel(self.test_file)['Solar']))
        with self.assertRaises(AttributeError):
            ts_data.non_existing_column
    
    def test_setattr(self):
        ts_data = TimeSeriesData('CAISO_net-load_2021.xlsx')
        ts_data.new_column = [0] * len(ts_data.data)
        self.assertListEqual(list(ts_data.new_column), [0] * len(ts_data.data))
    
    def test_summary(self):
        ts_data = TimeSeriesData('CAISO_net-load_2021.xlsx')
        summary = ts_data.summary()
        self.assertIn('Load', summary.columns)
        self.assertAlmostEqual(summary.loc['mean', 'Load'], pd.read_excel(self.test_file)['Load'].mean(), places=2)
    
    def test_get_column(self):
        ts_data = TimeSeriesData('CAISO_net-load_2021.xlsx')
        column_data = ts_data.get_column('Load', start_time_step=10, end_time_step=20)
        self.assertListEqual(list(column_data), list(pd.read_excel(self.test_file)['Load'][10:20]))
    
    def test_add_column(self):
        ts_data = TimeSeriesData('CAISO_net-load_2021.xlsx')
        new_data = pd.Series([1] * len(ts_data.data))
        ts_data.add_column('new_col', new_data)
        self.assertListEqual(list(ts_data.new_col), [1] * len(ts_data.data))

    
        
        
if __name__ == '__main__':
    unittest.main()