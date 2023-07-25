import kfp.dsl as dsl
import ml_pipeline_components

@dsl.pipeline(name='ML Pipeline', description='A pipeline for ML model training and prediction')
def ml_pipeline():
    # Load data
    load_data_task = ml_pipeline_components.load_data_component()

    # Preprocess data
    preprocess_task = ml_pipeline_components.preprocess_data_component(data=load_data_task.outputs)

    # Split data
    split_data_task = ml_pipeline_components.split_data_component(preprocessed_data=preprocess_task.output)

    # Train classifier with MLflow tracking
    train_classifier_task = ml_pipeline_components.train_classifier_component(
        X_train=split_data_task.outputs['X_train'], 
        y_train=split_data_task.outputs['y_train']
    )

    predictions_task = ml_pipeline_components.predict_test_data_component(model = train_classifier_task.output, X_test=split_data_task.outputs['X_test'])

    # Set the MLflow server URL
    mlflow_server_url = ml_pipeline_components.get_mlflow_server_url()

    # Log the MLflow server URL to the pipeline output
    dsl.ContainerOp(
        name='log-mlflow-server-url',
        image='python:3.8',
        command=['python', '-c', f'print("{mlflow_server_url}")']
    )
