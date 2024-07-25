from tfx.orchestration.pipeline import Pipeline
from tfx.orchestration.metadata import sqlite_metadata_connection_config
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.dsl.components.base import executor_spec

# Components
from components.data_ingestion import create_example_gen
from components.data_validation import create_data_validation
from components.data_transformation import create_transform
from components.model_trainer import create_trainer
from components.model_evaluator_and_pusher import create_evaluator_and_pusher

def create_pipeline(pipeline_name: str, pipeline_root: str, data_path: str,
                    serving_model_dir: str, module_file: str, project: str,
                    region: str):
    example_gen = create_example_gen(data_path)
    statistics_gen, schema_gen, example_validator = create_data_validation(example_gen)
    transform = create_transform(example_gen, schema_gen)
    
    # Define custom configurations for AI Platform Trainer to always use GPU
    custom_config = {
        ai_platform_trainer_executor.TRAINING_ARGS_KEY: {
            "scaleTier": "CUSTOM",
            "masterType": "m1-ultramem-40",
            "acceleratorConfig": {
                "count": 1,
                "type": "NVIDIA_TESLA_T4"  # Adjust according to your GPU requirement
            }
        }
    }
    
    trainer = create_trainer(transform, schema_gen, module_file, custom_config)
    evaluator, pusher, resolver = create_evaluator_and_pusher(example_gen, trainer, serving_model_dir)

    return Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=sqlite_metadata_connection_config(pipeline_root + '/metadata.db'),
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            resolver,
            evaluator,
            pusher
        ],
        enable_cache=True,
        beam_pipeline_args=[
            '--runner=DataflowRunner',
            '--project=' + project,
            '--temp_location=' + pipeline_root + '/tmp',
            '--region=' + region,
        ]
    )