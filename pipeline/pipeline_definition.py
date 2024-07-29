# pipeline/pipeline_definition.py

from tfx.orchestration import pipeline
from components.data_ingestion import create_example_gen
from components.data_validation import create_data_validation
from components.data_transformation import create_transform
from components.model_trainer import create_trainer
from components.model_evaluator_and_pusher import create_evaluator_and_pusher

def create_pipeline(pipeline_name: str, pipeline_root: str, data_path: str, serving_model_dir: str, module_file: str, project: str, region: str):
    query = """
    SELECT
        CAST(trip_seconds AS BIGNUMERIC) AS TripSeconds,
        CAST(trip_miles AS BIGNUMERIC) AS TripMiles,
        CAST(pickup_community_area AS STRING) AS PickupCommunityArea,
        CAST(dropoff_community_area AS STRING) AS DropoffCommunityArea,
        CAST(trip_start_timestamp AS STRING) AS TripStartTimestamp,
        CAST(trip_end_timestamp AS STRING) AS TripEndTimestamp,
        CAST(payment_type AS STRING) AS PaymentType,
        CAST(company AS STRING) AS Company
    FROM
        `bigquery-public-data.chicago_taxi_trips.taxi_trips`
    WHERE trip_seconds IS NOT NULL
       AND trip_miles IS NOT NULL
       AND pickup_community_area IS NOT NULL
       AND dropoff_community_area IS NOT NULL
       AND trip_start_timestamp IS NOT NULL
       AND trip_end_timestamp IS NOT NULL
       AND payment_type IS NOT NULL
       AND company IS NOT NULL
       AND trip_seconds != 0
       AND trip_miles != 0
       AND pickup_community_area != 0
       AND dropoff_community_area != 0;
    """
    example_gen = create_example_gen(query)
    statistics_gen, schema_gen, example_validator = create_data_validation(example_gen)
    transform = create_transform(example_gen, schema_gen)
    trainer = create_trainer(transform, schema_gen, module_file)

    # Use the modified function
    evaluator, pusher, resolver = create_evaluator_and_pusher(example_gen, trainer, serving_model_dir)

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        pusher
    ]

    if resolver is not None:
        components.append(resolver)
    if evaluator is not None:
        components.append(evaluator)

    pipeline_args = [
        '--project=' + project,
        '--runner=DataflowRunner',
        '--temp_location=gs://dataset_bucket_demo1/temp',
        '--region=' + region
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        beam_pipeline_args=pipeline_args,
        components=components
    )