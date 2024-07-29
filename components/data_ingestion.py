from tfx.v1.extensions.google_cloud_big_query import BigQueryExampleGen
from tfx.orchestration import pipeline
from tfx.orchestration.experimental import KubeflowV2DagRunner
from tfx.orchestration.experimental import KubeflowV2DagRunnerConfig

def create_example_gen(query: str):
    return BigQueryExampleGen(query=query)
    