import tensorflow as tf
import tensorflow_transform as tft

_FEATURE_KEYS = [
    "Trip Seconds", "Trip Miles", "Fare", "Pickup Community Area",
    "Dropoff Community Area", "Trip Start Timestamp", "Trip End Timestamp",
    "Payment Type", "Company"
]

_LABEL_KEY = 'Fare'

def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    outputs = {}

    for key in _FEATURE_KEYS:
        # Delete any rows that contain missing values, only numerical values allowed.
        transformed_feature = tft.scale_to_z_score(tft.coders.ExampleProtoCoder(inputs.schema).required(inputs[key]))
        outputs[key] = tf.where(tf.math.is_nan(transformed_feature), tf.constant(0, dtype=transformed_feature.dtype), transformed_feature)
    
    return outputs

def create_transform(example_gen, schema_gen):
    return Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        # Provide the preprocessing module file path correctly.
        module_file="components/data_transformation.py",
        splits_config=transform_pb2.SplitsConfig(
            analyze=['train'],
            transform=['train', 'eval']
        )
    )