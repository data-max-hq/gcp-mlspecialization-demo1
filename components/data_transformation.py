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

def t_name(key):
    """
    Rename the feature keys so that they don't clash with the raw keys when running the Evaluator component.
    Args:
      key: The original feature key
    Returns:
      key with '_xf' appended
    """
    return key + '_xf'

def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    outputs = {}

    # Pass through numeric features
    for key in _NUMERIC_FEATURES:
        outputs[t_name(key)] = inputs[key]

    # Pass through categorical numerical features
    for key in _CATEGORICAL_NUMERICAL_FEATURES:
        outputs[t_name(key)] = inputs[key]

    # Pass through categorical string features
    for key in _CATEGORICAL_STRING_FEATURES:
        outputs[t_name(key)] = inputs[key]

    # Pass through the label key
    outputs[t_name(_LABEL_KEY)] = inputs[_LABEL_KEY]
    
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