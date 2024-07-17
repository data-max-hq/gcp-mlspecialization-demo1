from tfx.components import Evaluator, Pusher, Trainer, Transform
from tfx.proto import pusher_pb2, evaluator_pb2, trainer_pb2
from tfx import v1 as tfx
import tensorflow_model_analysis as tfma
import tensorflow as tf
from tensorflow_transform import TFTransformOutput
import os
import dotenv

dotenv.load_dotenv()

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

_LABEL_KEY = 'Fare'
_FEATURE_KEYS = ["Trip Seconds", "Trip Miles", "Pickup Community Area", "Dropoff Community Area", "Trip Start Timestamp", "Trip End Timestamp", "Payment Type", "Company"]

def _get_tf_examples_serving_signature(model, tf_transform_output):
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop(_LABEL_KEY)
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)

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
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn

def input_fn(file_pattern, tf_transform_output, batch_size=200):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
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
    inputs = {key: tf.keras.layers.Input(name=key, shape=(), dtype=tf.float32) for key in feature_spec.keys()}
    
    x = tf.keras.layers.Concatenate()(list(inputs.values()))
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=inputs, outputs=output)

def run_fn(fn_args):
    tf_transform_output = TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    model = _build_keras_model(tf_transform_output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=1000,
        decay_rate=0.9)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])

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
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=200)
    )