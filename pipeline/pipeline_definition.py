import tfx
from tfx.orchestration import pipeline
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from components.data_ingestion import create_example_gen
from components.data_validation import create_data_validation
from components.data_transformation import create_transform
from components.model_trainer import create_trainer
from components.model_evaluator_and_pusher import create_evaluator_and_pusher

def create_pipeline(pipeline_name: str, pipeline_root: str, data_path: str, serving_model_dir: str, module_file: str, project: str, region: str):
    example_gen = create_example_gen(data_path)
    statistics_gen, schema_gen, example_validator = create_data_validation(example_gen)

    transform = create_transform(example_gen, schema_gen)
    
    trainer = create_trainer(transform, schema_gen, module_file)

    # Custom configurations dictionary for each component for GCP
    ai_platform_training_args = {
        'region': region,
        'project': project,
        'jobDir': f'gs://dataset_bucket_demo1/{pipeline_name}',
        'scaleTier': 'CUSTOM',
        'masterType':  'n1-highmem-16'
    }
    
    evaluator, pusher, resolver = create_evaluator_and_pusher(example_gen, trainer, serving_model_dir)
    
    # Use the GCP-specific trainer and pusher executors
    trainer.executor_spec = tfx.dsl.components.executor_spec.ExecutorClassSpec(ai_platform_trainer_executor.GenericExecutor)
    pusher.executor_spec = tfx.dsl.components.executor_spec.ExecutorClassSpec(ai_platform_pusher_executor.Executor)
    
    # Pass the custom configurations to the components
    transform.custom_config = ai_platform_training_args
    trainer.custom_config = ai_platform_training_args

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
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
        ]
    )