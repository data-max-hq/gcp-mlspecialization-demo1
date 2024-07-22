from tfx.components import Transform
import tensorflow as tf
import tensorflow_transform as tft
from tfx.proto import transform_pb2

# Define feature keys and label key
_FEATURE_KEYS = ["Age", "City_Category", "Gender", "Marital_Status", "Occupation", "Product_Category_1",
                 'Product_Category_2', 'Product_Category_3', "Stay_In_Current_City_Years"]
_CATEGORICAL_NUMERICAL_FEATURES = ["Marital_Status", "Occupation", "Product_Category_1", "Product_Category_2", "Product_Category_3"]
_CATEGORICAL_STRING_FEATURES = ["City_Category", "Age", "Stay_In_Current_City_Years", "Gender"]
_LABEL_KEY = 'Purchase'

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

def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    outputs = {}
    
    # Pass through categorical numerical features without transformation
    for key in _CATEGORICAL_NUMERICAL_FEATURES:
        outputs[t_name(key)] = inputs[key]
       
    # Pass through categorical string features without transformation
    for key in _CATEGORICAL_STRING_FEATURES:
        outputs[t_name(key)] = inputs[key]

    # Scale the label key
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