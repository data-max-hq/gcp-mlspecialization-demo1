from tfx.components import Transform
import tensorflow as tf
import tensorflow_transform as tft
from tfx.proto import transform_pb2

# Define feature keys and label key
_NUMERICAL_FEATURES = ["TripMiles", "TripSeconds"]
_CATEGORICAL_NUMERICAL_FEATURES = ["PickupCommunityArea", "DropoffCommunityArea"]
_CATEGORICAL_STRING_FEATURES = ["TripStartTimestamp", "TripEndTimestamp", "PaymentType", "Company"]
_BUCKET_FEATURES = ["PickupCommunityArea", "DropoffCommunityArea"]
_LABEL_KEY = 'Fare'
_FEATURE_BUCKET_COUNT = 10
_VOCAB_SIZE = 1000
_OOV_SIZE = 1

# Utility function for renaming keys
def t_name(key):
    return key + '_xf'

def _make_one_hot(x, key):
    """Make a one-hot tensor to encode categorical features.
    Args:
        X: A dense tensor
        key: A string key for the feature in the input
    Returns:
        A dense one-hot tensor as a float list
    """
    integerized = tft.compute_and_apply_vocabulary(
        x,
        top_k=_VOCAB_SIZE,
        num_oov_buckets=_OOV_SIZE,
        vocab_filename=key, name=key)
    depth = tft.experimental.get_vocabulary_size_by_name(key) + _OOV_SIZE
    one_hot_encoded = tf.one_hot(
        integerized, depth=tf.cast(depth, tf.int32),
        on_value=1.0, off_value=0.0)
    return tf.reshape(one_hot_encoded, [-1, depth])

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}
    
    # Scale numerical features to z-score
    for key in _NUMERICAL_FEATURES:
        outputs[t_name(key)] = tft.scale_to_z_score(inputs[key], name=key)

    # Bucketize numerical features to categorical
    for key in _BUCKET_FEATURES:
        outputs[t_name(key)] = tf.cast(tft.bucketize(inputs[key], _FEATURE_BUCKET_COUNT, name=key), dtype=tf.float32)

    # Apply one-hot encoding to string categorical features
    for key in _CATEGORICAL_STRING_FEATURES:
        outputs[t_name(key)] = _make_one_hot(inputs[key], key)

    # Apply one-hot encoding to numerical categorical features after converting them to string
    for key in _CATEGORICAL_NUMERICAL_FEATURES:
        outputs[t_name(key)] = _make_one_hot(tf.strings.strip(tf.strings.as_string(inputs[key])), key)

    # Pass through the label key
    outputs[_LABEL_KEY] = inputs[_LABEL_KEY]

    return outputs

def create_transform(example_gen, schema_gen):
    return Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file="components/data_transformation.py",
        splits_config=transform_pb2.SplitsConfig(
            analyze=['train'],
            transform=['train', 'eval']
        )
    )