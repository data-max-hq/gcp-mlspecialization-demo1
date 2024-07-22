from tfx.components import Transform
import tensorflow as tf
import tensorflow_transform as tft
from tfx.proto import transform_pb2

_FEATURE_KEYS = [
    "TripSeconds", "TripMiles", "Fare", "PickupCommunityArea",
    "DropoffCommunityArea", "TripStartTimestamp", "TripEndTimestamp",
    "PaymentType", "Company"
]

_CATEGORICAL_STRING_FEATURES = [
    "TripStartTimestamp", "TripEndTimestamp", "PaymentType", "Company"
]

_CATEGORICAL_NUMERICAL_FEATURES = [
    "PickupCommunityArea", "DropoffCommunityArea"
]

_LABEL_KEY = 'Fare'

def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns directly excluding any rows with missing values."""
    outputs = {}
    
    # Initialize a mask that cumulatively checks for non-missing values across all features
    valid_row_mask = None
    for key in _FEATURE_KEYS:
        current_feature = inputs[key]
        
        # Check for missing values
        if current_feature.dtype == tf.string:
            # For string fields, treat empty strings as missing
            current_missing_mask = tf.not_equal(current_feature, '')
        else:
            # For numeric fields, treat NaN values as missing
            current_missing_mask = tf.logical_not(tf.math.is_nan(current_feature))

        if valid_row_mask is None:
            valid_rows_mask = current_missing_mask
        else:
            valid_rows_mask = tf.logical_and(valid_rows_mask, current_missing_mask)

    for key in _FEATURE_KEYS:
        current_feature = inputs[key]
        
        if key in _CATEGORICAL_STRING_FEATURES:
            # Convert string to numeric indices, apply vocabulary
            indexed = tft.compute_and_apply_vocabulary(current_feature)
        elif key in _CATEGORICAL_NUMERICAL_FEATURES:
            # Convert string representation of numbers and scale
            numeric_feature = tf.strings.to_number(current_feature, out_type=tf.float32)
            indexed = tft.scale_to_z_score(numeric_feature)
        else:
            # Scale other numerical features
            numeric_feature = tf.strings.to_number(current_feature, out_type=tf.float32)
            indexed = tft.scale_to_z_score(numeric_feature)

        # Apply the valid rows mask to each transformed feature
        outputs[key] = tf.boolean_mask(indexed, valid_rows_mask)
    
    return outputs

def create_transform(example_gen, schema_gen):
    return Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file='components/data_transformation.py',
        splits_config=transform_pb2.SplitsConfig(
            analyze=['train'],
            transform=['train', 'eval']
        )
    )