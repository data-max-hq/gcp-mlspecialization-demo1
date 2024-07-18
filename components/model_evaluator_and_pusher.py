from tfx.components import Evaluator, Pusher
from tfx.proto import pusher_pb2, evaluator_pb2
from tfx import v1 as tfx
import tensorflow_model_analysis as tfma
import os
import dotenv

dotenv.load_dotenv()

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
endpoint_name = os.getenv("SERVING_ENDPOINT_NAME")
region = os.getenv("GOOGLE_CLOUD_REGION")




eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name: 'eval' and
        # remove the label_key.
        tfma.ModelSpec(
            signature_name='serving_default',
            label_key='Purchase',
            preprocessing_function_names=['transform_features'],
            )
        ],
    slicing_specs=[
        tfma.SlicingSpec()
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='RootMeanSquaredError', threshold=tfma.MetricThreshold(value_threshold=tfma.GenericValueThreshold(upper_bound={'value': 100000000000000})))
                ])]
    )


vertex_serving_spec = {
      'project_id': project_id,
      'endpoint_name': endpoint_name,
      # Remaining argument is passed to aiplatform.Model.deploy()
      # See https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api#deploy_the_model
      # for the detail.
      #
      # Machine type is the compute resource to serve prediction requests.
      # See https://cloud.google.com/vertex-ai/docs/predictions/configure-compute#machine-types
      # for available machine types and acccerators.
      'machine_type': 'n1-standard-2',
  }

serving_image = "europe-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-cpu.2-13:latest"


def create_evaluator_and_pusher(example_gen, trainer, serving_model_dir):
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        eval_config=eval_config,
        example_splits=['test']
    )

    pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      custom_config={
          tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
              True,
          tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
              region,
          tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY:
              serving_image,
          tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY:
            vertex_serving_spec,
      })
    return evaluator, pusher