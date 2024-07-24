import tensorflow as tf
import tensorflow_transform as tft
from tfx.v1 import components
from tfx.proto import transform_pb2
from tfx.components import Transform

NUMERICAL_FEATURES = ['TripSeconds', 'TripMiles']
CATEGORICAL_NUMERICAL_FEATURES = ['PickupCommunityArea', 'DropoffCommunityArea']
CATEGORICAL_STRING_FEATURES = ['PaymentType', 'Company']
VOCAB_SIZE = 1000
OOV_SIZE = 10
LABEL_KEY = 'Fare'

def t_name(key):
    return key + '_xf'

def _make_one_hot(x, key):
    integerized = tft.compute_and_apply_vocabulary(x,
            top_k=VOCAB_SIZE,
            num_oov_buckets=OOV_SIZE,
            vocab_filename=key, name=key)
    depth = tft.experimental.get_vocabulary_size_by_name(key) + OOV_SIZE
    one_hot_encoded = tf.one_hot(
        integerized,
        depth=tf.cast(depth, tf.int32),
        on_value=1.0,
        off_value=0.0)
    return tf.reshape(one_hot_encoded, [-1, depth])

def preprocessing_fn(inputs):
    outputs = {}
    for key in NUMERICAL_FEATURES:
        outputs[t_name(key)] = tft.scale_to_z_score(inputs[key], name=key)
    for key in CATEGORICAL_STRING_FEATURES:
        outputs[t_name(key)] = _make_one_hot(inputs[key], key)
    for key in CATEGORICAL_NUMERICAL_FEATURES:
        outputs[t_name(key)] = _make_one_hot(tf.strings.strip(
            tf.strings.as_string(inputs[key])), key)
    outputs[LABEL_KEY] = inputs[LABEL_KEY]
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