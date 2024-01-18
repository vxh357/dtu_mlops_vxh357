# Script assumes you have sklearn v1.1.3 installed
# You can update to the newest version using
# pip install -U scikit-learn
# Also you need to have matplotlib installed
import click
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@click.group()
def main():
    """Main command group for the K-nearest neighbor classification script."""
    pass

@main.command()
@click.option('-o', '--output', type=str, help='Specify the filename to save the trained model.')
def train(output):
    """Load data, train the model, and save it.

    Args:
        output (str): The filename to save the trained model.

    Example:
        python main.py train -o 'model.ckpt'
    """
    n_neighbors = 1

    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    X = X[:, [0, 2]]

    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Save the model
    dump(pipeline, output)
    click.echo(f"Model saved to {output}")

@main.command()
@click.option('-i', '--input', type=str, help='Specify the trained model to load.')
@click.option('-d', '--datapoint', type=list, help='Specify a user-defined datapoint for inference.')
def infer(input, datapoint):
    """Load a trained model and run prediction on input data.

    Args:
        input (str): The filename of the trained model to load.
        datapoint (list): A user-defined datapoint for inference.

    Example:
        python main.py infer -i 'model.ckpt' -d [0, 1]
    """
    pipeline = load(input)
    prediction = pipeline.predict([datapoint])
    click.echo(f"Inference result: {prediction}")

@main.command()
@click.option('-i', '--input', type=str, help='Specify the trained model to load.')
@click.option('-o', '--output', type=str, help='Specify the filename to write the generated plot.')
def plot(input, output):
    """Load a trained model and construct the decision boundary plot.

    Args:
        input (str): The filename of the trained model to load.
        output (str): The filename to write the generated plot.

    Example:
        python main.py plot -i 'model.ckpt' -o 'generated_plot.png'
    """
    pipeline = load(input)
    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    X = X[:, [0, 2]]

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        pipeline,
        X,
        alpha=0.8,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        shading="auto",
    )
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=20)
    plt.savefig(output)
    click.echo(f"Decision boundary plot saved to {output}")

@main.command()
@click.option('-o', '--option', type=str, help='Specify an option to adjust hyperparameter optimization.')
def optim(option):
    """Load data, run hyperparameter optimization, and print optimal parameters.

    Args:
        option (str): An optional argument to adjust the hyperparameter optimization.

    Example:
        python main.py optim -o 'option'
    """
    n_neighbors = 1

    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    X = X[:, [0, 2]]

    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

    parameters = {"knn__n_neighbors": list(range(1, 10))}
    hyper_pipeline = GridSearchCV(Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=n_neighbors))]), parameters)
    hyper_pipeline.fit(X_train, y_train)
    click.echo(f"Optimal parameters: {hyper_pipeline.best_params_}")

if __name__ == '__main__':
    main()

