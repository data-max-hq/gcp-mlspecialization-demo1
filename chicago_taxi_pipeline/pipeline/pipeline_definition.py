from components.data_ingestion import create_example_gen
from components.data_transformation import create_transform
from components.data_validation import create_data_validation
from components.model_evaluator_and_pusher import create_evaluator_and_pusher
from components.model_trainer import create_trainer
from tfx.orchestration import pipeline


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    query: str,
    serving_model_dir: str,
    module_file: str,
    project: str,
    region: str,
):
    # Data ingestion component using BigQuery
    example_gen = create_example_gen(query)
    statistics_gen, schema_gen, example_validator = create_data_validation(example_gen)
    transform = create_transform(example_gen, schema_gen)
    trainer = create_trainer(transform, schema_gen, module_file)

    evaluator, pusher, resolver = create_evaluator_and_pusher(
        example_gen, trainer, serving_model_dir
    )

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        pusher,
    ]

    if resolver is not None:
        components.append(resolver)
    if evaluator is not None:
        components.append(evaluator)

    pipeline_args = [
        "--project=" + project,
        "--runner=DataflowRunner",
        "--temp_location=gs://dataset_bucket_demo1/temp",
        "--region=" + region,
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        beam_pipeline_args=pipeline_args,
        components=components,
    )
