from tfx import v1 as tfx
from pipeline.pipeline_definition import create_pipeline
import os
import dotenv

dotenv.load_dotenv()

PIPELINE_NAME = os.getenv("PIPELINE_NAME")
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
DATA_QUERY = os.getenv("DATA_QUERY")
MODULE_ROOT = os.getenv("MODULE_ROOT")
SERVING_MODEL_DIR = os.getenv("SERVING_MODEL_DIR")

PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'

runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
    config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
    output_filename=PIPELINE_DEFINITION_FILE)

_ = runner.run(
    create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        query=DATA_QUERY,
        module_file=f'{MODULE_ROOT}/model_trainer.py',
        serving_model_dir=SERVING_MODEL_DIR))