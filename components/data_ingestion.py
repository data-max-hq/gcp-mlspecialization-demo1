from tfx.proto import example_gen_pb2
from tfx.v1.proto import Output
from tfx.v1.proto import SplitConfig 
from tfx.components import BigQueryExampleGen

def create_example_gen(query: str):
    example_gen = BigQueryExampleGen(query=query)
    return example_gen