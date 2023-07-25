import kfp
import ml_pipeline_kubeflow

# Set your Kubeflow experiment and run names
experiment_name = 'my-ml-experiment'
run_name = 'my-ml-run'

# Initialize the Kubeflow client
client = kfp.Client()

# Compile the pipeline
pipeline_filename = 'ml_pipeline.yaml'
kfp.compiler.Compiler().compile(ml_pipeline_kubeflow.ml_pipeline, pipeline_filename)

# Create and run the pipeline
client.create_run_from_pipeline_func(ml_pipeline_kubeflow.ml_pipeline, experiment_name=experiment_name, run_name=run_name)
