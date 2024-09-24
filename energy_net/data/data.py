import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, List

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__))
DATASETS_DIRECTORY = os.path.join(DATA_DIRECTORY, 'datasets')

class TimeSeriesData:
    """
    TimeSeriesData class for handling time series data from CSV and Excel files.
    Attributes:
        file_path (str): Path to the data file.
        file_type (str): Type of the data file (csv or xlsx).
        start_time_step (int, optional): Starting time step for slicing the data.
        end_time_step (int, optional): Ending time step for slicing the data.
        data (pd.DataFrame): Loaded data as a pandas DataFrame.
    Methods:
        __init__(file_name: str, start_time_step: int = None, end_time_step: int = None):
            Initializes the TimeSeriesData object with the given file name and optional time steps.
        _get_file_type(file_name: str) -> str:
            Extracts the file type from the file name.
        _load_data() -> pd.DataFrame:
            Loads the data based on the file type.
        _slice_data(start: int, end: int) -> pd.DataFrame:
            Slices the data between start and end time steps.
        __getattr__(name: str):
            Dynamically gets an attribute if it exists in the data columns.
        __setattr__(name: str, value: Any):
            Dynamically sets an attribute if it is a predefined attribute, otherwise sets it in the data.
        summary():
            Returns a summary of the dataset.
        get_column(column_name: str, start_time_step: int = None, end_time_step: int = None):
            Gets a specific column with optional time step slicing.
        add_column(column_name: str, data: pd.Series):
            Adds a new column to the dataset.
        save(file_name: str):
            Saves the current state of the dataset to a CSV file.
        visualize(columns: List[str], start_time_step: int = None, end_time_step: int = None):
            Visualizes specified columns of the dataset.
    """
    def __init__(self, file_name: str, start_time_step: int = None, end_time_step: int = None):
        self.file_path = os.path.join(DATASETS_DIRECTORY, file_name)
        self.file_type = self._get_file_type(file_name)
        self.start_time_step = start_time_step
        self.end_time_step = end_time_step
        self.data = self._load_data()

        if start_time_step is not None or end_time_step is not None:
            self.data = self._slice_data(start_time_step, end_time_step)
    
    def _get_file_type(self, file_name: str) -> str:
        """Extract the file type from the file name."""
        _, file_extension = os.path.splitext(file_name)
        if file_extension in ['.csv', '.xlsx']:
            return file_extension.lstrip('.')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _load_data(self) -> pd.DataFrame:
        """Load the data based on the file type."""
        if self.file_type == 'csv':
            return pd.read_csv(self.file_path)
        elif self.file_type == 'xlsx':
            return pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
    
    def _slice_data(self, start: int, end: int) -> pd.DataFrame:
        """Slice the data between start and end time steps."""
        return self.data.iloc[start:end]
    
    def __getattr__(self, name: str):
        """Dynamically get an attribute."""
        if name in self.data.columns:
            return self.data[name]
        raise AttributeError(f"'TimeSeriesData' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any):
        """Dynamically set an attribute."""
        if name in ['file_path', 'file_type', 'start_time_step', 'end_time_step', 'data']:
            super().__setattr__(name, value)
        else:
            self.data[name] = value

    def summary(self):
        """Get a summary of the dataset."""
        return self.data.describe()
    
    def get_column(self, column_name: str, start_time_step: int = None, end_time_step: int = None):
        """Get a specific column with optional time step slicing."""
        if column_name not in self.data.columns:
            raise AttributeError(f"Column '{column_name}' not found in the dataset.")
        data = self.data[column_name]
        if start_time_step is not None or end_time_step is not None:
            data = data.iloc[start_time_step:end_time_step]
        return data
    
    def add_column(self, column_name: str, data: pd.Series):
        """Add a new column to the dataset."""
        self.data[column_name] = data
    
    def save(self, file_name: str):
        """Save the current state of the dataset to a CSV file."""
        save_path = os.path.join(DATASETS_DIRECTORY, file_name)
        self.data.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")

    def visualize(self, columns: List[str], start_time_step: int = None, end_time_step: int = None):
        """Visualize specified columns of the dataset."""
        if start_time_step is not None or end_time_step is not None:
            data = self._slice_data(start_time_step, end_time_step)
        else:
            data = self.data
        
        for column in columns:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in the dataset.")
            plt.plot(data['Date'], data[column], label=column)
        
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.title('Time Series Data Visualization')
        plt.legend()
        plt.show()