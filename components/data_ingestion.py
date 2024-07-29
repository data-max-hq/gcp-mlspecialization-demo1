from tfx.components import QueryBasedExampleGen


def create_example_gen(query: str):
    """
    Create a BigQueryExampleGen component.
    
    Args:
        query (str): The SQL query to extract data from BigQuery.

    Returns:
        BigQueryExampleGen: An instance of BigQueryExampleGen initialized with the given query.
    """
    # Create the BigQueryExampleGen component with the provided query
    example_gen = QueryBasedExampleGen(query=query)
    return example_gen