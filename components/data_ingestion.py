from tfx.components import BigQueryExampleGen

def create_example_gen(query: str):
    example_gen = BigQueryExampleGen(query=query)
    return example_gen