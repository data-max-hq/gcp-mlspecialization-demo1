from tfx import v1 as tfx
from pipeline.pipeline_definition import create_pipeline
import os
import dotenv

dotenv.load_dotenv()

PROJECT_NAME = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")
PIPELINE_NAME = os.getenv("PIPELINE_NAME")
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
MODULE_ROOT = os.getenv("MODULE_ROOT")
SERVING_MODEL_DIR = os.getenv("SERVING_MODEL_DIR")

# Define your BigQuery query here
QUERY = """
SELECT
    CAST(trip_seconds AS FLOAT64) AS TripSeconds,
    CAST(trip_miles AS FLOAT64) AS TripMiles,
    CAST(pickup_community_area AS STRING) AS PickupCommunityArea,
    CAST(dropoff_community_area AS STRING) AS DropoffCommunityArea,
    CAST(trip_start_timestamp AS STRING) AS TripStartTimestamp,
    CAST(trip_end_timestamp AS STRING) AS TripEndTimestamp,
    CAST(payment_type AS STRING) AS PaymentType,
    CAST(company AS STRING) AS Company
  FROM
    `bigquery-public-data.chicago_taxi_trips.taxi_trips`
  WHERE
    trip_seconds IS NOT NULL AND
    trip_miles IS NOT NULL AND
    pickup_community_area IS NOT NULL AND
    dropoff_community_area IS NOT NULL AND
    trip_start_timestamp IS NOT NULL AND
    trip_end_timestamp IS NOT NULL AND
    payment_type IS NOT NULL AND
    company IS NOT NULL AND
    trip_miles != 0 AND
    pickup_community_area != 0 AND
    dropoff_community_area != 0
"""

PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'

runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
    config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
    output_filename=PIPELINE_DEFINITION_FILE)

_ = runner.run(
    create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        query=QUERY,  # Pass the query here
        serving_model_dir=SERVING_MODEL_DIR,
        module_file=f'{MODULE_ROOT}/model_trainer.py',
        project=PROJECT_NAME,
        region=GOOGLE_CLOUD_REGION
    )
)