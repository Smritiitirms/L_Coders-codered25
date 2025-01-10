import pandas as pd

class SanityCheckAgent:
    def run(self, data: pd.DataFrame):
        """Perform sanity checks on the dataset."""
        report = {}
        
        # Missing Values
        report['missing_values'] = data.isnull().sum().to_dict()

        # Data Types
        report['data_types'] = data.dtypes.apply(str).to_dict()

        # Shape of the dataset
        report['shape'] = data.shape

        # Duplicate Rows
        report['duplicates'] = data.duplicated().sum()

        return report
