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

vertex_serving_spec = {
    'project_id': project_id,
    'endpoint_name': endpoint_name,
    'machine_type': 'n1-standard-2',
}

serving_image = "us-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-cpu.2-13:latest"

def create_evaluator_and_pusher(example_gen, trainer, serving_model_dir):
    # Check if there's a previous model
    resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
                'latest_blessed_model_resolver')

    evaluator_needed = False
    try:
        resolver.outputs['model'].get()[0]
        evaluator_needed = True
    except IndexError:
        print("No previous model found. Skipping evaluation.")

    if evaluator_needed:
        eval_config = tfma.EvalConfig(
            model_specs=[
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
                        tfma.MetricConfig(class_name='RootMeanSquaredError', threshold=tfma.MetricThreshold(value_threshold=tfma.GenericValueThreshold(upper_bound={'value': 10000000})))
                    ]
                )
            ]
        )

        evaluator = Evaluator(
            examples=example_gen.outputs['examples'],
            model=trainer.outputs['model'],
            baseline_model=resolver.outputs['model'],
            eval_config=eval_config,
            example_splits=['test']
        )

        pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
            model=trainer.outputs['model'],
            model_blessing=evaluator.outputs['blessing'],
            custom_config={
                tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
                tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
                tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY: serving_image,
                tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY: vertex_serving_spec,
            }
        )
        return evaluator, pusher, resolver

    pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
        model=trainer.outputs['model'],
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
            tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY: serving_image,
            tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY: vertex_serving_spec,
        }
    )
    return None, pusher, None