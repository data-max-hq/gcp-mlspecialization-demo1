from tfx.components import Transform
import tensorflow as tf
import tensorflow_transform as tft
from tfx.proto import transform_pb2

# Define feature keys and label key
_FEATURE_KEYS = [
    "TripSeconds", "TripMiles", "PickupCommunityArea", "DropoffCommunityArea", 
    "TripStartTimestamp", "TripEndTimestamp", "PaymentType", "Company"
]
_CATEGORICAL_NUMERICAL_FEATURES = ["PickupCommunityArea", "DropoffCommunityArea"]
_CATEGORICAL_STRING_FEATURES = ["TripStartTimestamp", "TripEndTimestamp", "PaymentType", "Company"]
_NUMERIC_FEATURES = ["TripSeconds", "TripMiles"]
_LABEL_KEY = 'Fare'
_VOCAB_SIZE = 1000
_OOV_SIZE = 10

def t_name(key):
    """
    Rename the feature keys so that they don't clash with the raw keys when
    running the Evaluator component.
    Args:
      key: The original feature key
    Returns:
      key with '_xf' appended
    """
    return key + '_xf'


def _make_one_hot(x, key):
    """Make a one-hot tensor to encode categorical features.
    Args:
      x: A dense tensor
      key: A string key for the feature in the input
    Returns:
      A dense one-hot tensor as a float list
    """
    integerized = tft.compute_and_apply_vocabulary(x,
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE,
            vocab_filename=key, name=key)
    depth = (
        tft.experimental.get_vocabulary_size_by_name(key) + _OOV_SIZE)
    one_hot_encoded = tf.one_hot(
        integerized,
        depth=tf.cast(depth, tf.int32),
        on_value=1.0,
        off_value=0.0)
    return tf.reshape(one_hot_encoded, [-1, depth])

def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    outputs = {}

    # Scale numeric features
    for key in _NUMERIC_FEATURES:
        outputs[t_name(key)] = tft.scale_to_z_score(inputs[key])

    # One-hot encode categorical numerical features
    for key in _CATEGORICAL_NUMERICAL_FEATURES:
        outputs[t_name(key)] = _make_one_hot(inputs[key], key)

    # One-hot encode categorical string features
    for key in _CATEGORICAL_STRING_FEATURES:
        outputs[t_name(key)] = _make_one_hot(inputs[key], key)

    # Scale the label key
    outputs[t_name(_LABEL_KEY)] = tft.scale_to_z_score(inputs[_LABEL_KEY])
    
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