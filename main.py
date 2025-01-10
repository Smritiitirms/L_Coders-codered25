import streamlit as st
import os
import pandas as pd
import plotly.express as px
import sweetviz as sv
from datetime import datetime
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from agents.sanity_check import SanityCheckAgent
from agents.summary import SummaryAgent
from agents.report import ReportAgent
from transformers import pipeline
import shap

# Set up folders
UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

sanity_check = SanityCheckAgent()
summary_agent = SummaryAgent()
report_agent = ReportAgent()

# Initialize SQLite database for file history
DB_PATH = "file_history.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create a table for storing file history if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS file_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        upload_date TEXT
    )
""")
conn.commit()

# Streamlit App Configuration
st.set_page_config(
    page_title="Synapse AI: Medical Data Analysis",
    layout="wide",
    menu_items={"About": "An AI-Powered Medical Data Analysis Tool"},
)

# Sidebar for navigation
st.sidebar.title("🔍 Medical Data Explorer")
page = st.sidebar.radio("Navigation", ["Home", "Feature Analysis", "Diagnostics", "Report","Assistant","About"])


import requests

def generate_text_description(file_path):
    """
    Send a dataset to the local GPT server for summary generation.
    
    Args:
        file_path (str): Path to the dataset file to be analyzed.
    
    Returns:
        str: Summary text returned by the local GPT server.
    """
    try:
        # Define the server endpoint
        url = "http://localhost:5000/generate_summary"  # Update this to match your local server endpoint
        files = {"file": open(file_path, "rb")}

        # Make the request to the server
        response = requests.post(url, files=files)

        # Check the response status
        if response.status_code == 200:
            return response.json().get("summary", "No summary available.")
        else:
            return f"Error: Server responded with status code {response.status_code}."
    except Exception as e:
        return f"Error: {str(e)}"

# Global variable for loaded data
uploaded_file = None
data = None

# Function to load the dataset
def load_data(file):
    file_type = file.name.split(".")[-1].lower()
    filepath = os.path.join(UPLOAD_FOLDER, file.name)

    with open(filepath, "wb") as f:
        f.write(file.getbuffer())

    if file_type == "csv":
        return pd.read_csv(filepath)
    elif file_type == "xlsx":
        return pd.read_excel(filepath)
    elif file_type == "json":
        return pd.read_json(filepath)
    elif file_type == "parquet":
        return pd.read_parquet(filepath)
    else:
        st.error("Unsupported file format.")
        return None

# Function to log uploaded file to the database
def log_file_history(file_name):
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO file_history (file_name, upload_date) VALUES (?, ?)", (file_name, upload_date))
    conn.commit()

# Function to fetch file history from the database
def get_file_history():
    cursor.execute("SELECT file_name, upload_date FROM file_history ORDER BY upload_date DESC")
    return cursor.fetchall()

if page == "Home":
    st.title("📊 Synapse AI")
    st.subheader("Upload and Explore Medical Datasets")

    uploaded_file = st.file_uploader(
        "Upload a file (CSV, Excel, JSON, Parquet)",
        type=["csv", "xlsx", "json", "parquet"],
        accept_multiple_files=False,
    )

    st.subheader("🗂 File Upload History")
    history = get_file_history()
    if history:
        for file_name, upload_date in history:
            st.write(f"📄 {file_name} (Uploaded on: {upload_date})")
        if st.button("Clear File Upload History"):
            cursor.execute("DELETE FROM file_history")
            conn.commit()
            st.success("File upload history cleared!")
    else:
        st.info("No files uploaded yet.")

    if st.button("Load Data"):
        if uploaded_file:
            data = load_data(uploaded_file)
            if data is not None:
                st.session_state["data"] = data
                st.success("File uploaded successfully!")
                st.dataframe(data.head())
                log_file_history(uploaded_file.name)
        else:
            st.warning("Please upload a file.")

elif page == "Diagnostics":
    st.title("🔬 AI-Driven Diagnostics")
    st.subheader("Predictive Analysis for Medical Data")

    if "data" in st.session_state and st.session_state["data"] is not None:
        data = st.session_state["data"]

        target_column = st.selectbox("Select Target Column (e.g., Diagnosis/Outcome):", data.columns)
        feature_columns = st.multiselect("Select Features for Prediction:", [col for col in data.columns if col != target_column])

        # Recommendation model selection (moved from sidebar)
        st.subheader("📌 Recommended Models")
        recomended_models = ['Logistic Regression', 'Random Forest', 'XGBoost Classifier', 'Gradient Boosting Classifier']
        pred_model = st.selectbox("Choose a Model for Prediction", recomended_models)

        if st.button("Run Predictive Analysis"):
            if len(feature_columns) > 0:
                # Train-Test Split
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import classification_report
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from xgboost import XGBClassifier

                X = data[feature_columns]
                y = data[target_column]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Load the selected model
                def load_pred_model(model_name):
                    if model_name == 'Logistic Regression':
                        model = LogisticRegression()
                    elif model_name == 'Random Forest': 
                        model = RandomForestClassifier()
                    elif model_name == 'XGBoost Classifier':
                        model = XGBClassifier()
                    elif model_name == 'Gradient Boosting Classifier':
                        model = GradientBoostingClassifier()
                    else:
                        raise ValueError(f"Unknown model name: {model_name}")
                    return model

                # Initialize and train the selected model
                model = load_pred_model(pred_model)
                model.fit(X_train, y_train)

                # Predictions and evaluation
                predictions = model.predict(X_test)
                report = classification_report(y_test, predictions, output_dict=True)

                # Visual presentation of the report in a readable format
                st.markdown("### Classification Report:")
                st.markdown(f"#### Model: {pred_model}")
                
                # Show results as text in a clean and readable format
                for label, metrics in report.items():
                    if label != 'accuracy':
                        st.markdown(f"{label}")
                        st.markdown(f"Precision: {metrics['precision']:.2f}")
                        st.markdown(f"Recall: {metrics['recall']:.2f}")
                        st.markdown(f"F1-Score: {metrics['f1-score']:.2f}")
                        st.markdown(f"Support: {metrics['support']}")
                        st.markdown("---")

                st.markdown(f"*Accuracy*: {report['accuracy']:.2f}")
                
                # Optional: You can also display confusion matrix, precision-recall curves, etc.
            else:
                st.warning("Please select features for prediction.")
    else:
        st.warning("Upload data first!")

elif page == "Feature Analysis":
    st.title("📈 Feature Analysis")
    if "data" in st.session_state and st.session_state["data"] is not None:
        data = st.session_state["data"]
        
        # Overview
        st.subheader("🔍 Dataset Overview")
        st.write(f"Number of Rows: {data.shape[0]}")
        st.write(f"Number of Columns: {data.shape[1]}")
        st.dataframe(data.head())

        # Visualization Type Selection
        st.subheader("📊 Smart Visualizations")
        vis_type = st.radio("Choose Visualization Type", ["Distribution", "Correlation Matrix", "Scatter Plot", "Box Plot"])

        if vis_type == "Distribution":
            feature_column = st.selectbox("Select a Feature to Analyze", options=data.columns)
            if pd.api.types.is_numeric_dtype(data[feature_column]):
                fig = px.histogram(data, x=feature_column, title=f"Distribution of {feature_column}")
            else:
                fig = px.bar(
                    data[feature_column].value_counts().reset_index(),
                    x="index",
                    y=feature_column,
                    title=f"Value Counts of {feature_column}",
                    labels={"index": feature_column, feature_column: "Count"}
                )
            st.plotly_chart(fig)

        elif vis_type == "Correlation Matrix":
            st.write("Correlation Matrix for Numeric Features:")
            corr_matrix = data.corr()
            fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix", color_continuous_scale="Viridis")
            st.plotly_chart(fig)

        elif vis_type == "Scatter Plot":
            x_column = st.selectbox("Select X-axis Feature", options=data.columns)
            y_column = st.selectbox("Select Y-axis Feature", options=data.columns)
            fig = px.scatter(data, x=x_column, y=y_column, title=f"{y_column} vs {x_column}")
            st.plotly_chart(fig)

        elif vis_type == "Box Plot":
            feature_column = st.selectbox("Select a Feature to Analyze", options=data.columns)
            fig = px.box(data, y=feature_column, title=f"Box Plot of {feature_column}")
            st.plotly_chart(fig)

        # Sweetviz Advanced Report Generation
        st.subheader("📄 Generate Sanity Check Report")
        if st.button("Generate Sweetviz Report"):
            with st.spinner("Generating report..."):
                report = sv.analyze(data)
                report_path = os.path.join(REPORT_FOLDER, "sweetviz_report.html")
                report.show_html(filepath=report_path, open_browser=False)
            
            with open(report_path, "rb") as file:
                st.download_button("Download Sweetviz Report", file, file_name="sweetviz_report.html")
    else:
        st.warning("No data available! Please upload a dataset on the Home page.")

elif page == "Report":
    st.title("📄 Automated Detailed Report")

    if "data" in st.session_state and st.session_state["data"] is not None:
        data = st.session_state["data"]

        st.subheader("📊 Data Overview")
        st.write(f"Rows: {data.shape[0]}")
        st.write(f"Columns: {data.shape[1]}")
        st.dataframe(data.head())

       

        st.subheader("🧠 Explainable AI Insights")
        if st.button("Generate Explainability Report"):
            with st.spinner("Generating explainability insights..."):
        # Select the target column for model prediction
                target_column = st.selectbox("Select Target Column:", data.columns)
                feature_columns = [col for col in data.columns if col != target_column]

        # Prepare the feature set (X) and target set (y)
            X = data[feature_columns].select_dtypes(include=["number"]).dropna()
            y = data[target_column]

        # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train a RandomForest model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

        # Use SHAP to explain the model predictions
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

        # Display SHAP values and feature importance
            st.write("### Feature Importance (SHAP Summary Plot):")
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')

            st.write("### Detailed SHAP Values for Test Set:")
            st.write(shap_values)

        # Optionally, display a heatmap for feature importance
            shap_df = pd.DataFrame(shap_values[0], columns=X_test.columns)
            plt.figure(figsize=(10, 8))
            sns.heatmap(shap_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=1)
            st.pyplot(bbox_inches='tight')


    else:
        st.warning("No data available! Please upload a dataset on the Home page.")


elif page == "Assistant":
    st.title("\U0001F5E3 AI Assistant")
    st.subheader("Interact with the AI assistant to analyze your uploaded data or explore the reference information about diabetes.")

    # Load reference file on diabetes in women
    reference_file_path = "diabetes_women_reference.txt"
    if os.path.exists(reference_file_path):
        with open(reference_file_path, "r") as ref_file:
            reference_data = ref_file.read()
    else:
        st.error("Reference file on diabetes in women not found.")
        reference_data = None

    # Display uploaded data preview if available
    if "data" in st.session_state and st.session_state["data"] is not None:
        data = st.session_state["data"]
        st.write("Uploaded File Preview:")
        st.dataframe(data.head())

    # User question input
    user_input = st.text_area("Ask a question about your uploaded data or explore the diabetes reference:")

    if st.button("Get Response"):
        if user_input:
            # Prepare prompt with dataset summary and user input
            prompt = ""
            if "data" in st.session_state and st.session_state["data"] is not None:
                summary = data.describe(include="all").to_string()
                first_rows = data.head(5).to_string(index=False)
                prompt += (
                    f"The dataset contains the following summary:\n" + summary + "\n\n"
                    f"Here are the first 5 rows of the dataset:\n" + first_rows + "\n\n"
                )

            if reference_data:
                prompt += "Reference information about diabetes in women:\n" + reference_data + "\n\n"

            prompt += "User's question:\n" + user_input

            # Generate assistant response (assuming generate_assistant_response is implemented)
            try:
                response = generate_assistant_response(prompt)
                st.write("AI Assistant Response:")
                st.write(response)
            except Exception as e:
                st.error("Error generating response: " + str(e))
        else:
            st.warning("Please enter a question to get a response.")
    else:
        st.info("Waiting for your input.")   
elif page == "About":
    st.title("About Synapse AI")
    st.subheader("An AI-Powered Medical Data Analysis Tool")

    st.write("""
    *Synapse AI* is an innovative tool designed to provide deep insights and analysis for medical datasets. Built using cutting-edge technologies such as Python, Streamlit, and AI-driven algorithms, Synapse AI empowers healthcare professionals, researchers, and data analysts to uncover patterns, predict outcomes, and generate comprehensive reports with ease.

    Our tool simplifies the process of analyzing medical data, making it accessible and user-friendly for users of all skill levels. Whether you're working with patient records, clinical trials, or diagnostic data, Synapse AI helps you make informed decisions through its powerful suite of features, including:
    - *Sanity checks* for data integrity
    - *Predictive analysis* using advanced machine learning models
    - *Comprehensive visualizations* for data exploration
    - *Automated report generation* for easy sharing and presentation

    *Why Synapse AI?*
    - *User-Friendly Interface:* Designed with healthcare professionals in mind, offering an intuitive and easy-to-navigate experience.
    - *AI-Powered Insights:* Utilize advanced machine learning models for predictive analytics and diagnostic predictions.
    - *Real-Time Reporting:* Automatically generate Sweetviz and Pandas Profiling reports to provide a thorough analysis of the data.
    - *Interactive Visualizations:* Quickly visualize data trends with dynamic charts and plots.

    Our goal is to assist in making healthcare data analysis more efficient, accurate, and accessible for everyone.

    *Contact Us:*
    - If you have any questions, suggestions, or would like to collaborate, feel free to reach out to us.
    """)
