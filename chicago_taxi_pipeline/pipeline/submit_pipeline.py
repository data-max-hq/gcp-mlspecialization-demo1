# docs_infra: no_execute
import logging
import os

import dotenv
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

logging.getLogger().setLevel(logging.INFO)


dotenv.load_dotenv()

PIPELINE_NAME = os.getenv("PIPELINE_NAME")
PIPELINE_DEFINITION_FILE = PIPELINE_NAME + "_pipeline.json"
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")

aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)

job = pipeline_jobs.PipelineJob(
    template_path=PIPELINE_DEFINITION_FILE, display_name=PIPELINE_NAME
)

job.submit()
