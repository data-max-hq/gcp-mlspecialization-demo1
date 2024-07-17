from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import tensorflow as tf
import dotenv
import os


dotenv.load_dotenv()

PROJECT_NAME = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
API_ENDPOINT = os.getenv("API_ENDPOINT")

# Updated feature list to match your new dataset
def serialize_example(input_data):
    feature = {
        'Trip Seconds': tf.train.Feature(int64_list=tf.train.Int64List(value=[input_data['Trip Seconds']])),
        'Trip Miles': tf.train.Feature(float_list=tf.train.FloatList(value=[input_data['Trip Miles']])),
        'Fare': tf.train.Feature(float_list=tf.train.FloatList(value=[input_data['Fare']])),
        'Pickup Community Area': tf.train.Feature(int64_list=tf.train.Int64List(value=[input_data['Pickup Community Area']])),
        'Dropoff Community Area': tf.train.Feature(int64_list=tf.train.Int64List(value=[input_data['Dropoff Community Area']])),
        'Trip Start Timestamp': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_data['Trip Start Timestamp'].encode()])),
        'Trip End Timestamp': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_data['Trip End Timestamp'].encode()])),
        'Payment Type': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_data['Payment Type'].encode()])),
        'Company': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_data['Company'].encode()])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = GOOGLE_CLOUD_REGION,
    api_endpoint: str = API_ENDPOINT,
):
    """
    `instances` can be either a single instance of type dict or a list
    of instances.
    """
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]

    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("Response:")
    print("  deployed_model_id:", response.deployed_model_id)
    predictions = response.predictions
    for prediction in predictions:
        print("  prediction:", prediction)

# Example usage
project = GOOGLE_CLOUD_PROJECT
endpoint_id = ENDPOINT_ID
location = GOOGLE_CLOUD_REGION
api_endpoint = API_ENDPOINT

input_data = {
    'Trip Seconds': 300,
    'Trip Miles': 5.5,
    'Fare': 15.0,
    'Pickup Community Area': 8,
    'Dropoff Community Area': 32,
    'Trip Start Timestamp': '2023-01-01T00:00:00Z',
    'Trip End Timestamp': '2023-01-01T00:05:00Z',
    'Payment Type': 'Credit Card',
    'Company': 'Flash Cab',
}

serialized_example = serialize_example(input_data)
encoded_example = tf.io.encode_base64(serialized_example).numpy().decode('utf-8')

predict_custom_trained_model_sample(
    project=project,
    endpoint_id=endpoint_id,
    location=location,
    instances={"examples": {"b64": encoded_example}}
)