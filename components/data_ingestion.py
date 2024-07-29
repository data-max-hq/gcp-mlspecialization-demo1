from tfx.extensions.google_cloud_big_query.big_query_example_gen.component import BigQueryExampleGen

def create_example_gen(query: str):
    """
    Create a BigQueryExampleGen component.
    
    Args:
        query (str): The SQL query to extract data from BigQuery.

    Returns:
        BigQueryExampleGen: An instance of BigQueryExampleGen initialized with the given query.
    """
    # Create the BigQueryExampleGen component with the provided query
    example_gen = BigQueryExampleGen(query=query)
    return example_gen