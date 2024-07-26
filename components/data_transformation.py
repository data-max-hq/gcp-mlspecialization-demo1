import tensorflow as tf
import tensorflow_transform as tft
from tfx.v1 import components
from tfx.proto import transform_pb2
from tfx.components import Transform

# Feature definitions
NUMERICAL_FEATURES = ['TripSeconds', 'TripMiles']
CATEGORICAL_NUMERICAL_FEATURES = ['PickupCommunityArea', 'DropoffCommunityArea']
CATEGORICAL_STRING_FEATURES = ['TripStartTimestamp', 'TripEndTimestamp', 'PaymentType', 'Company']
LABEL_KEY = 'Fare'
VOCAB_SIZE = 1000
OOV_SIZE = 10

def t_name(key):
    """Append '_xf' to the transformed feature name to avoid clashing with raw features."""
    return key + '_xf'

def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x

  default_value = '' if x.dtype == tf.string else -1
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)


def _make_one_hot(x, key):
  """Make a one-hot tensor to encode categorical features.
  Args:
    X: A dense tensor
    key: A string key for the feature in the input
  Returns:
    A dense one-hot tensor as a float list
  """
  integerized = tft.compute_and_apply_vocabulary(x,
          top_k=VOCAB_SIZE,
          num_oov_buckets=OOV_SIZE,
          vocab_filename=key, name=key)
  depth = (
      tft.experimental.get_vocabulary_size_by_name(key) + OOV_SIZE)
  one_hot_encoded = tf.one_hot(
      integerized,
      depth=tf.cast(depth, tf.int32),
      on_value=1.0,
      off_value=0.0)
  return tf.reshape(one_hot_encoded, [-1, depth])

def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    outputs = {}
    
    # Process numerical features and cast to float
    for key in NUMERICAL_FEATURES:
        outputs[t_name(key)] = inputs[key]
    
    # Process categorical string features and cast to string
    for key in CATEGORICAL_STRING_FEATURES:
        outputs[t_name(key)] = _make_one_hot(inputs[key], key)
    
    # Process categorical numerical features and cast to int
    for key in CATEGORICAL_NUMERICAL_FEATURES:
        outputs[t_name(key)] = _make_one_hot(tf.strings.strip(
        tf.strings.as_string(inputs[key])))
    
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