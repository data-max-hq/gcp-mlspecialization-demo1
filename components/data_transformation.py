from tfx.components import Transform
import tensorflow as tf
import tensorflow_transform as tft
from tfx.proto import transform_pb2

# Feature keys
_FEATURE_KEYS = [
    "Trip Seconds",
    "Trip Miles",
    "Fare",
    "Pickup Community Area",
    "Dropoff Community Area",
    "Trip Start Timestamp",
    "Trip End Timestamp",
    "Payment Type",
    "Company"
]

# Categorical features
_CATEGORICAL_NUMERICAL_FEATURES = [
    "Pickup Community Area",
    "Dropoff Community Area"
]

# String-based categorical features
_CATEGORICAL_STRING_FEATURES = [
    "Trip Start Timestamp",
    "Trip End Timestamp",
    "Payment Type",
    "Company"
]

# Label key
_LABEL_KEY = 'Fare'
_VOCAB_SIZE = 1000
_OOV_SIZE = 10

def t_name(key):
    """Rename the feature keys to avoid clashes with raw keys."""
    return key + '_xf'

def _fill_in_missing(x):
    """Replace missing values in a SparseTensor."""
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = '' if x.dtype == tf.string else -1
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)

def _make_one_hot(x, key):
    """Make a one-hot tensor to encode categorical features."""
    integerized = tft.compute_and_apply_vocabulary(
        x,
        top_k=_VOCAB_SIZE,
        num_oov_buckets=_OOV_SIZE,
        vocab_filename=key, name=key)

    depth = tft.experimental.get_vocabulary_size_by_name(key) + _OOV_SIZE
    one_hot_encoded = tf.one_hot(
        integerized,
        depth=tf.cast(depth, tf.int32),
        on_value=1.0,
        off_value=0.0)
    return tf.reshape(one_hot_encoded, [-1, depth])

def preprocessing_fn(inputs):
    outputs = {}

    for key in _CATEGORICAL_NUMERICAL_FEATURES:
        outputs[t_name(key)] = _make_one_hot(
            tf.strings.strip(tf.strings.as_string(_fill_in_missing(inputs[key]))), key)
       
    for key in _CATEGORICAL_STRING_FEATURES:
        outputs[t_name(key)] = _make_one_hot(_fill_in_missing(inputs[key]), key)

    # Process numerical features
    for key in {"Trip Seconds", "Trip Miles"}:
        outputs[t_name(key)] = tft.scale_to_z_score(_fill_in_missing(inputs[key]))

    # Process numerical features excluding label
    for key in set(_FEATURE_KEYS) - set(_CATEGORICAL_STRING_FEATURES) - {_LABEL_KEY} - {"Trip Seconds", "Trip Miles"}:
        outputs[t_name(key)] = tft.scale_to_z_score(_fill_in_missing(inputs[key]))

    # Process the label
    outputs[_LABEL_KEY] = tft.scale_to_z_score(inputs[_LABEL_KEY])

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