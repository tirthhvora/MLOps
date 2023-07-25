import kfp.dsl as dsl
import mlflow

def get_mlflow_server_url():
    # Replace 'your_mlflow_service_name' with the name of the MLflow service in your Kubernetes cluster
    mlflow_service_name = 'localhost:5000'

    # Retrieve the MLflow server URL using the Kubernetes API
    service_url = mlflow.get_service_url(mlflow_service_name)
    return service_url


@dsl.component
def load_data_component():
    return dsl.ContainerOp(
        name='load-data',
        image='python:3.8',
        command=['python', 'load_data_script.py'],
        output_artifact_paths={'data': '/data'},
    )

@dsl.component
def preprocess_data_component(data):
    return dsl.ContainerOp(
        name='preprocess-data',
        image='python:3.8',
        command=['python', 'preprocessing_script.py'],
        arguments=['--data', data],
        output_artifact_paths={'preprocessed_data': '/data'},
    )

@dsl.component
def split_data_component(final_df):
    return dsl.ContainerOp(
        name='split-data',
        image='python:3.8',
        command=['python', 'data_split_script.py'],
        arguments=['--final_df', final_df],
        output_artifact_paths={
            'X_train': '/data/X_train.npy',
            'X_test': '/data/X_test.npy',
            'y_train': '/data/y_train.npy',
            'y_test': '/data/y_test.npy',
        },
    )

@dsl.component
def train_classifier_component(X_train, y_train):
    return dsl.ContainerOp(
        name='train-classifier',
        image='python:3.8',
        command=['python', 'model_building_script.py'],
        arguments=[
            '--X_train', X_train,
            '--y_train', y_train
        ],
        output_artifact_paths={'model': '/data/model.pkl'},
    )

@dsl.component
def predict_test_data_component(model, X_test):
    return dsl.ContainerOp(
        name='predict-test-data',
        image='python:3.8',
        command=['python', 'prediction_script.py'],
        arguments=[
            '--model', model,
            '--X_test', X_test
        ],
        output_artifact_paths={'y_pred': '/data/y_pred.npy'},
    )
