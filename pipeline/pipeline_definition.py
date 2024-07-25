from tfx.orchestration import pipeline

from components.data_ingestion import create_example_gen
from components.data_validation import create_data_validation
from components.data_transformation import create_transform
from components.model_trainer import create_trainer
from components.model_evaluator_and_pusher import create_evaluator_and_pusher

def create_pipeline(pipeline_name: str, pipeline_root: str, data_path: str, serving_model_dir: str, module_file: str, project: str, region: str):
    example_gen = create_example_gen(data_path)
    statistics_gen, schema_gen, example_validator = create_data_validation(example_gen)
    transform = create_transform(
    example_gen.outputs['examples'], 
    schema_gen.outputs['schema'], 
    custom_config={
        'ai_platform_training_args': {
            'project': project,
            'region': region,
            'machineType': 'n1-highmem-32',  # Specifying machine type for Transform component
        }
    }
)

    # Configure the trainer component with custom_config for AI Platform Training.
    trainer = create_trainer(
        transform.outputs['transformed_examples'], 
        schema_gen.outputs['schema'], 
        module_file, 
        custom_config={
            'ai_platform_training_args': {
                'project': project,
                'region': region,
                'masterType': 'n1-highmem-32',  # Specify machine type here for training
            }
        }
    )

    evaluator, pusher, resolver = create_evaluator_and_pusher(example_gen, trainer, serving_model_dir)

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