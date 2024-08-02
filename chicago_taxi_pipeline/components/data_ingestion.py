from tfx.v1.extensions.google_cloud_big_query import BigQueryExampleGen


def create_example_gen(query: str):
    return BigQueryExampleGen(query=query)
