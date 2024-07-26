import tensorflow as tf
import tensorflow_transform as tft
from tfx.v1 import components
from tfx.proto import transform_pb2
from tfx.components import Transform

# Feature definitions
NUMERICAL_FEATURES = ['TripSeconds', 'TripMiles']
CATEGORICAL_NUMERICAL_FEATURES = ['PickupCommunityArea', 'DropoffCommunityArea']
CATEGORICAL_STRING_FEATURES = ['TripStartTimestamp', 'TripEndTimestamp', 'PaymentType', 'Company']
VOCAB_SIZE = 1000
OOV_SIZE = 10
LABEL_KEY = 'Fare'

def t_name(key):
    """Append '_xf' to the transformed feature name to avoid clashing with raw features."""
    return key + '_xf'

def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    outputs = {}
    
    # Process numerical features and cast to float
    for key in NUMERICAL_FEATURES:
        outputs[key] = tf.cast(inputs[key], tf.float32)
    
    # Process categorical string features and cast to string
    for key in CATEGORICAL_STRING_FEATURES:
        outputs[key] = tf.cast(inputs[key], tf.string)
    
    # Process categorical numerical features and cast to int
    for key in CATEGORICAL_NUMERICAL_FEATURES:
        outputs[key] = tf.cast(inputs[key], tf.int64)
    
    # Pass through the label and cast to float (assuming Fare is a numerical feature)
    outputs[LABEL_KEY] = tf.cast(inputs[LABEL_KEY], tf.float32)
    
    return outputs

def create_transform(example_gen, schema_gen):
    return Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file="components/data_transformation.py",
        splits_config=transform_pb2.SplitsConfig(
            analyze=['train'],
            transform=['train', 'eval']
        ),
    )