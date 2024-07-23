from tfx.components import Trainer
from tfx.proto import trainer_pb2
from tfx import v1 as tfx
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import TFTransformOutput
from tfx_bsl.public import tfxio

from absl import logging

import os
import dotenv

dotenv.load_dotenv()

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

_LABEL_KEY = 'Fare'
_FEATURE_KEYS = [
    "TripSeconds", "TripMiles", "PickupCommunityArea", "DropoffCommunityArea",
    "TripStartTimestamp", "TripEndTimestamp", "PaymentType", "Company"
]

# Define a function to get transformed feature names
def tft_name(key: str) -> str:
    return key + '_xf'

_TRANSFORM_FEATURE_KEYS = [tft_name(key) for key in _FEATURE_KEYS]

def _get_tf_examples_serving_signature(model, tf_transform_output):
    """Returns a serving signature that accepts `tf.Example`."""
    
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        """Returns the output for serving."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()

        # Remove label feature
        raw_feature_spec.pop(_LABEL_KEY)
        required_feature_spec = {k: v for k, v in raw_feature_spec.items() if k in _FEATURE_KEYS}

        raw_features = tf.io.parse_example(serialized_tf_example, required_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)

        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn

def _get_transform_features_signature(model, tf_transform_output):
    """Returns a serving signature that applies tf.Transform to features."""
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed features."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn

def input_fn(file_pattern, tf_transform_output, batch_size=200):
    raw_feature_spec = tf_transform_output.raw_feature_spec().copy()
    print("Raw feature spec:", raw_feature_spec)

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=raw_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type='GZIP'),
        label_key=_LABEL_KEY
    )
    print("Dataset element spec:", dataset.element_spec)
    
    for example in dataset.take(1):
        print("Example in dataset:", example)

    return dataset

def export_serving_model(tf_transform_output, model, output_dir):
    """Exports a keras model for serving."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    signatures = {
        'serving_default': _get_tf_examples_serving_signature(model, tf_transform_output),
        'transform_features': _get_transform_features_signature(model, tf_transform_output),
    }

    model.save(output_dir, save_format='tf', signatures=signatures)

def _build_keras_model(tf_transform_output: TFTransformOutput) -> tf.keras.Model:
    """Creates a DNN Keras model for predicting fare amount."""
    
    # Use raw feature spec from transformed output
    feature_spec = tf_transform_output.raw_feature_spec().copy()
    feature_spec = {k: v for k, v in feature_spec.items() if k in _TRANSFORM_FEATURE_KEYS}
    feature_spec.pop(_LABEL_KEY, None)

    inputs = {}
    for key, spec in feature_spec.items():
        if isinstance(spec, tf.io.VarLenFeature):
            inputs[key] = tf.keras.layers.Input(
                shape=[None], name=key, dtype=spec.dtype, sparse=True)
        elif isinstance(spec, tf.io.FixedLenFeature):
            inputs[key] = tf.keras.layers.Input(
                shape=spec.shape or [1], name=key, dtype=spec.dtype)
        else:
            raise ValueError('Spec type is not supported: ', key, spec)
          
    x = tf.keras.layers.Concatenate()(tf.nest.flatten(inputs))
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=inputs, outputs=output)

def run_fn(fn_args):
    """Train the model based on given args."""
    tf_transform_output = TFTransformOutput(fn_args.transform_output)
    
    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    model = _build_keras_model(tf_transform_output)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-1,
        decay_steps=1000,
        decay_rate=0.9)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mean_squared_error',  # Using MSE for regression
        metrics=['mean_absolute_error']  # MAE as an additional metric
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')
    
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, early_stopping])

    # Ensure the transformation layer is saved with the model
    export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)

def create_trainer(transform, schema_gen, module_file):
    return Trainer(
        module_file=module_file,
        custom_config={
            'ai_platform_training_args': {
                'project': GOOGLE_CLOUD_PROJECT,
                'region': GOOGLE_CLOUD_REGION,
                'job-dir': f'{GCS_BUCKET_NAME}/jobs'
            },
            'pre_transform_stats': transform.outputs['pre_transform_stats']
        },
        transformed_examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=50000),
        eval_args=trainer_pb2.EvalArgs(num_steps=10000),
    )