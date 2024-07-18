import os
import dotenv
from tfx.orchestration import pipeline
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from components.data_ingestion import create_example_gen
from components.data_validation import create_data_validation
from components.data_transformation import create_transform
from components.model_trainer import create_trainer
from components.model_evaluator_and_pusher import create_evaluator_and_pusher

# Load environment variables from .env file
dotenv.load_dotenv()

# Environment variables
PIPELINE_NAME = os.getenv('PIPELINE_NAME')
PIPELINE_ROOT = os.getenv('PIPELINE_ROOT')
DATA_PATH = os.getenv('DATA_PATH')
SERVING_MODEL_DIR = os.getenv('SERVING_MODEL_DIR')
MODULE_FILE = os.getenv('MODULE_FILE')
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
GOOGLE_CLOUD_REGION = os.getenv('GOOGLE_CLOUD_REGION')
METADATA_PATH = os.getenv('METADATA_PATH')

def create_pipeline(pipeline_name: str, pipeline_root: str, query: str, data_path: str, serving_model_dir: str, module_file: str, project: str, region: str):
    example_gen = create_example_gen(query)
    statistics_gen, schema_gen, example_validator = create_data_validation(example_gen)
    transform = create_transform(example_gen, schema_gen)
    trainer = create_trainer(transform, schema_gen, module_file)
    evaluator, pusher = create_evaluator_and_pusher(example_gen, trainer, serving_model_dir)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            evaluator,
            pusher
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH)
    )


if __name__ == '__main__':
    BeamDagRunner().run(
        create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=DATA_PATH,
            serving_model_dir=SERVING_MODEL_DIR,
            module_file=MODULE_FILE,
            project=GOOGLE_CLOUD_PROJECT,
            region=GOOGLE_CLOUD_REGION
        )
    )