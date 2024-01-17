import pickle
from datetime import datetime
from typing import List

from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse
from sklearn import datasets

app = FastAPI()

# Define class labels for predictions
classes = ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]

# Load the pre-trained machine learning model from a pickle file
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.post("/iris_v1/")
def iris_inference_v1(
    sepal_length: float, sepal_width: float, petal_length: float, petal_width: float
):
    """
    Endpoint for performing version 1 of the iris flower species inference.

    Parameters:
    - sepal_length (float): Sepal length of the iris flower.
    - sepal_width (float): Sepal width of the iris flower.
    - petal_length (float): Petal length of the iris flower.
    - petal_width (float): Petal width of the iris flower.

    Returns:
    - dict: A dictionary containing the predicted class and its corresponding integer representation.
    """
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = prediction.item()
    return {"prediction": classes[prediction], "prediction_int": prediction}


# Initialize a prediction database CSV file and write the header
with open("prediction_database.csv", "w") as file:
    file.write("time, sepal_length, sepal_width, petal_length, petal_width, prediction\n")


def add_to_database(
    now: str,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    prediction: int,
):
    """
    Function to add a prediction record to the prediction database CSV file.

    Parameters:
    - now (str): Current timestamp in string format.
    - sepal_length (float): Sepal length of the iris flower.
    - sepal_width (float): Sepal width of the iris flower.
    - petal_length (float): Petal length of the iris flower.
    - petal_width (float): Petal width of the iris flower.
    - prediction (int): Integer representation of the predicted iris flower class.
    """
    with open("prediction_database.csv", "a") as file:
        file.write(
            f"{now}, {sepal_length}, {sepal_width}, {petal_length}, {petal_width}, {prediction}\n"
        )


@app.post("/iris_v2/")
async def iris_inference_v2(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    background_tasks: BackgroundTasks,
):
    """
    Endpoint for performing version 2 of the iris flower species inference and adding the prediction to the database.

    Parameters:
    - sepal_length (float): Sepal length of the iris flower.
    - sepal_width (float): Sepal width of the iris flower.
    - petal_length (float): Petal length of the iris flower.
    - petal_width (float): Petal width of the iris flower.
    - background_tasks (BackgroundTasks): FastAPI background tasks for asynchronous processing.
    """
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = prediction.item()

    now = str(datetime.now())
    background_tasks.add_task(
        add_to_database,
        now,
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
        prediction,
    )

    return {"prediction": classes[prediction], "prediction_int": prediction}


@app.get("/iris_monitoring/", response_class=HTMLResponse)
async def iris_monitoring():
    """
    Endpoint for generating a monitoring report of data drift, data quality, and target drift for the Iris dataset.

    Returns:
    - HTMLResponse: An HTML response containing the monitoring report.
    """
    # Load the Iris dataset
    iris_frame = datasets.load_iris(as_frame="True").frame

    # Create a monitoring report with specified metrics
    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ]
    )

    # Run the monitoring report on the current and reference data
    data_drift_report.run(
        current_data=iris_frame.iloc[:60],
        reference_data=iris_frame.iloc[60:],
        column_mapping=None,
    )
    
    # Save the monitoring report as an HTML file
    data_drift_report.save_html("monitoring.html")

    # Read the generated HTML report
    with open("monitoring.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)
