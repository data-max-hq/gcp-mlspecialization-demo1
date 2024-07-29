from tfx.components import BigQueryExampleGen
from tfx.proto import example_gen_pb2
from tfx.v1.proto import Output
from tfx.v1.proto import SplitConfig

def create_example_gen(query: str):
    # Define output config specifying the splits
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1),
            example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=1)
        ])
    )

    # Create the BigQueryExampleGen component
    example_gen = BigQueryExampleGen(
        query=query,
        output_config=output_config
    )

    return example_gen