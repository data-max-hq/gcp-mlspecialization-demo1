from tfx.components import Pusher
from tfx import v1 as tfx
import os
import dotenv

dotenv.load_dotenv()

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
endpoint_name = os.getenv("SERVING_ENDPOINT_NAME")
region = os.getenv("GOOGLE_CLOUD_REGION")

vertex_serving_spec = {
    'project_id': project_id,
    'endpoint_name': endpoint_name,
    # Remaining argument is passed to aiplatform.Model.deploy()
    # See https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api#deploy_the_model
    # for the details.
    #
    # Machine type is the compute resource to serve prediction requests.
    # See https://cloud.google.com/vertex-ai/docs/predictions/configure-compute#machine-types
    # for available machine types and accelerators.
    'machine_type': 'n1-standard-2',
}

serving_image = "us-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-cpu.2-13:latest"

# Function modified to only include Pusher without evaluation
def create_pusher_for_new_model(trainer, serving_model_dir):
    pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
        model=trainer.outputs['model'],
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
            tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY: serving_image,
            tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY: vertex_serving_spec,
        }
    )
    return pusher