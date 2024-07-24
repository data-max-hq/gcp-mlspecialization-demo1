from tfx.components import Trainer
from tfx.proto import trainer_pb2
from tfx import v1 as tfx
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import TFTransformOutput
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2, anomalies_pb2, statistics_pb2
import gcsfs

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
_TRANSFORM_FEATURE_KEYS = [
    "TripSeconds_xf", "TripMiles_xf", "PickupCommunityArea_xf", 
    "DropoffCommunityArea_xf", "TripStartTimestamp_xf", "TripEndTimestamp_xf", 
    "PaymentType_xf", "Company_xf"
]

def _get_tf_examples_serving_signature(model, tf_transform_output):
    """Returns a serving signature that accepts `tensorflow.Example`."""
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop(_LABEL_KEY)
        required_feature_spec = {
            k: v for k, v in raw_feature_spec.items() if k in _FEATURE_KEYS
        }
        raw_features = tf.io.parse_example(serialized_tf_example, required_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn

def _get_transform_features_signature(model, tf_transform_output):
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        return transformed_features

    return transform_features_fn

def input_fn(file_pattern, tf_transform_output, batch_size=200):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type='GZIP'),
        label_key=_LABEL_KEY
    )
    return dataset

def export_serving_model(tf_transform_output, model, output_dir):
    model.tft_layer = tf_transform_output.transform_features_layer()
    signatures = {
        'serving_default': _get_tf_examples_serving_signature(model, tf_transform_output),
        'transform_features': _get_transform_features_signature(model, tf_transform_output),
    }
    model.save(output_dir, save_format='tf', signatures=signatures)

def _build_keras_model(tf_transform_output: TFTransformOutput) -> tf.keras.Model:
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(_LABEL_KEY)

    numerical_inputs = {}
    categorical_inputs = {}

    # Define inputs and separate them into numerical and categorical
    for key, spec in feature_spec.items():
        if isinstance(spec, tf.io.FixedLenFeature):
            if spec.dtype == tf.float32:
                numerical_inputs[key] = tf.keras.layers.Input(shape=[1], name=key, dtype=spec.dtype)
            elif spec.dtype == tf.string:
                categorical_inputs[key] = tf.keras.layers.Input(shape=[1], name=key, dtype=spec.dtype)
        else:
            raise ValueError('Unsupported feature type: ', key, spec)

    # Process numerical features
    concatenated_numerical_features = tf.keras.layers.Concatenate()(list(numerical_inputs.values()))

    # Process categorical features
    def encode_string_inputs(inputs):
        encoded_layers = []
        for input_name, input_tensor in inputs.items():
            vocab_size = 1000  # Adjust according to the feature
            embedding_dim = 16
            num_oov_buckets = 1

            lookup = tf.keras.layers.StringLookup(
                vocabulary=None,
                num_oov_indices=num_oov_buckets,
                mask_token=None,
                output_mode="int",
                name=f"{input_name}_lookup"
            )
            encoded_layer = lookup(input_tensor)
            embedding = tf.keras.layers.Embedding(vocab_size + num_oov_buckets, embedding_dim)(encoded_layer)
            encoded_layer = tf.keras.layers.Reshape(target_shape=(embedding_dim,))(embedding)
            encoded_layers.append(encoded_layer)
        return encoded_layers

    # Encode and concatenate categorical features
    encoded_categorical_features = encode_string_inputs(categorical_inputs)
    concatenated_categorical_features = tf.keras.layers.Concatenate()(encoded_categorical_features)

    # Combine all features
    concatenated_features = tf.keras.layers.Concatenate()([concatenated_numerical_features, concatenated_categorical_features])

    # Build the rest of the model
    x = tf.keras.layers.Dense(512, activation='relu')(concatenated_features)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs={**numerical_inputs, **categorical_inputs}, outputs=output)

def run_fn(fn_args):
    tf_transform_output = TFTransformOutput(fn_args.transform_output)
    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)
    model = _build_keras_model(tf_transform_output)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-1,
        decay_steps=1000,
        decay_rate=0.9
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch'
    )
   
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, early_stopping]
    )
   
    export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)

def create_trainer(transform, schema_gen, module_file):
    return Trainer(
        module_file=module_file,
        custom_config={
            'ai_platform_training_args': {
                'project': GOOGLE_CLOUD_PROJECT,
                'region': GOOGLE_CLOUD_REGION,
                'job-dir': f'{GCS_BUCKET_NAME}/jobs'
            }
        },
        transformed_examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=500),
        eval_args=trainer_pb2.EvalArgs(num_steps=100),
    )