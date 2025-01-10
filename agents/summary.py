import pandas as pd
import sweetviz as sv

class SummaryAgent:
    def run(self, data: pd.DataFrame):
        """
        Generate an enhanced summary of the dataset using Sweetviz.
        This creates an HTML report containing detailed EDA.
        """
        # Create a Sweetviz report
        report = sv.analyze(data)
        
        # Define the path for the generated report
        report_path = "dataset_summary.html"
        
        # Save the report
        report.show_html(report_path)
        
        return report_path
