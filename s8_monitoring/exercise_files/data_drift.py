from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_suite import TestSuite
from sklearn import datasets

# Load the Iris dataset as a DataFrame
iris_frame = datasets.load_iris(as_frame="auto").frame

# Create a data drift report
data_drift_report = Report(
    metrics=[
        DataDriftPreset(),      # Calculate data drift metrics
        DataQualityPreset(),    # Calculate data quality metrics
        TargetDriftPreset(),    # Calculate target drift metrics
    ]
)

# Run the data drift report with current and reference data
data_drift_report.run(
    current_data=iris_frame.iloc[:60],    # Data for the current period
    reference_data=iris_frame.iloc[60:],  # Data for the reference period
    column_mapping=None,                  # Optional: Mapping of columns if needed
)

# Save the data drift report as an HTML file
data_drift_report.save_html("test.html")

# Create a data stability test suite
data_stability = TestSuite(
    tests=[
        DataStabilityTestPreset(),    # Run data stability tests
    ]
)

# Run the data stability tests with current and reference data
data_stability.run(
    current_data=iris_frame.iloc[:60],    # Data for the current period
    reference_data=iris_frame.iloc[60:],  # Data for the reference period
    column_mapping=None,                  # Optional: Mapping of columns if needed
)

# Save the data stability test results as an HTML file
data_stability.save_html("test2.html")
