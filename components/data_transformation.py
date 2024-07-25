import tensorflow as tf
import tensorflow_transform as tft
from tfx.v1 import components
from tfx.proto import transform_pb2
from tfx.components import Transform
import gc  # Import garbage collection module

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

def _make_one_hot(x, key):
    """Make a one-hot tensor to encode categorical features."""
    integerized = tft.compute_and_apply_vocabulary(
        x,
        top_k=VOCAB_SIZE,
        num_oov_buckets=OOV_SIZE,
        vocab_filename=key,
        name=key
    )
    depth = tft.experimental.get_vocabulary_size_by_name(key) + OOV_SIZE
    one_hot_encoded = tf.one_hot(
        integerized,
        depth=tf.cast(depth, tf.int32),
        on_value=1.0,
        off_value=0.0
    )
    return tf.reshape(one_hot_encoded, [-1, depth])

def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    outputs = {}
    
    # Process numerical features
    for key in NUMERICAL_FEATURES:
        outputs[t_name(key)] = tft.scale_to_z_score(inputs[key], name=key)
    
    # Process categorical string features
    for key in CATEGORICAL_STRING_FEATURES:
        outputs[t_name(key)] = _make_one_hot(inputs[key], key)
    
    # Process categorical numerical features
    for key in CATEGORICAL_NUMERICAL_FEATURES:
        outputs[t_name(key)] = _make_one_hot(tf.strings.strip(
            tf.strings.as_string(inputs[key])), key
        )
    
    # Pass through the label
    outputs[LABEL_KEY] = inputs[LABEL_KEY]
    
    return outputs

def clear_session():
    """Clear TensorFlow session to free memory."""
    tf.keras.backend.clear_session()
    gc.collect()

def transform_shard(example_gen, schema_gen, shard_num, shard_size):
    """Process a shard of the data."""
    # Adjust your module_file to read and process only the specific shard of data
    module_file = "components/data_transformation.py"

    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file,
        splits_config=transform_pb2.SplitsConfig(
            analyze=['train'],
            transform=['train', 'eval']
        )
    )

    # Perform cleanup to free memory
    clear_session()

def create_transform(example_gen, schema_gen, total_data_size=100000, shard_size=10000):
    """Create the TFX Transform component with data sharding."""
    for shard_num in range(0, total_data_size, shard_size):
        transform_shard(example_gen, schema_gen, shard_num, shard_size)