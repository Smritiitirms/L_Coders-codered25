from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

class ReportAgent:
    def generate_pdf_report(self, data, target_column=None, include_summary=True, include_visualizations=True, save_path="report.pdf"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Add Title Page
        pdf.add_page()
        pdf.set_font("Arial", size=16, style="B")
        pdf.cell(0, 10, "Data Analysis Report", ln=True, align="C")
        pdf.ln(10)

        # Add Summary Section
        if include_summary:
            pdf.set_font("Arial", size=12, style="B")
            pdf.cell(0, 10, "Summary", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.ln(5)

            # Add dataset overview
            summary = f"""
            Number of Rows: {data.shape[0]}
            Number of Columns: {data.shape[1]}
            """
            pdf.multi_cell(0, 10, summary)
            pdf.ln(5)

        # Add Visualizations
        if include_visualizations:
            if not os.path.exists("temp_charts"):
                os.makedirs("temp_charts")

            # Distribution of the target column
            if target_column:
                plt.figure(figsize=(8, 5))
                sns.countplot(data[target_column])
                plt.title(f"Distribution of {target_column}")
                chart_path = f"temp_charts/{target_column}_distribution.png"
                plt.savefig(chart_path)
                plt.close()
                pdf.add_page()
                pdf.cell(0, 10, f"Visualization: Distribution of {target_column}", ln=True)
                pdf.image(chart_path, x=10, y=30, w=180)

            # Correlation heatmap
            numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
            if len(numeric_columns) > 1:
                plt.figure(figsize=(8, 5))
                sns.heatmap(data[numeric_columns].corr(), annot=True, cmap="coolwarm")
                plt.title("Correlation Heatmap")
                chart_path = "temp_charts/correlation_heatmap.png"
                plt.savefig(chart_path)
                plt.close()
                pdf.add_page()
                pdf.cell(0, 10, "Visualization: Correlation Heatmap", ln=True)
                pdf.image(chart_path, x=10, y=30, w=180)

            # Cleanup temp charts
            for file in os.listdir("temp_charts"):
                os.remove(os.path.join("temp_charts", file))
            os.rmdir("temp_charts")

        # Save PDF
        pdf.output(save_path)
