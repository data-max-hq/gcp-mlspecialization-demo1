from tfx.components import BigQueryExampleGen
from tfx.proto import example_gen_pb2
from tfx.v1.proto import Output
from tfx.v1.proto import SplitConfig 

def create_example_gen(query: str):
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1),
            example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=1)
        ])
    )
    
    example_gen = BigQueryExampleGen(query=query, output_config=output_config)
    
    return example_gen

# Your BigQuery SQL query
QUERY = """
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

example_gen = create_example_gen(QUERY)