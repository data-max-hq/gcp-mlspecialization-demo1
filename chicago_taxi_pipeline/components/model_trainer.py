import os

import dotenv
import tensorflow as tf
from absl import logging
from tensorflow_transform import TFTransformOutput
from tfx.components import Trainer
from tfx.proto import trainer_pb2

dotenv.load_dotenv()

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
_BATCH_SIZE = 32

_LABEL_KEY = "Fare"
_FEATURE_KEYS = [
    "TripSeconds",
    "TripMiles",
    "PickupCommunityArea",
    "DropoffCommunityArea",
    "TripStartTimestamp",
    "TripEndTimestamp",
    "PaymentType",
    "Company",
]


# Function to provide serving signature
def _get_tf_examples_serving_signature(model, tf_transform_output):
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")]
    )
    def serve_tf_examples_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop(_LABEL_KEY)
        required_feature_spec = {
            k: v for k, v in raw_feature_spec.items() if k in _FEATURE_KEYS
        }

        raw_features = tf.io.parse_example(serialized_tf_example, required_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info("serve_transformed_features = %s", transformed_features)
        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


# Function to provide transform features signature
def _get_transform_features_signature(model, tf_transform_output):
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")]
    )
    def transform_features_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info("eval_transformed_features = %s", transformed_features)
        return transformed_features

    return transform_features_fn


# Function to create dataset from TFRecord files
def input_fn(file_pattern, tf_transform_output, batch_size=200):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(
            filenames, compression_type="GZIP"
        ),
        label_key=_LABEL_KEY,
    )
    print("Dataset element spec:", dataset.element_spec)

    return dataset


# Function to export the model
def export_serving_model(tf_transform_output, model, output_dir):
    model.tft_layer = tf_transform_output.transform_features_layer()

    signatures = {
        "serving_default": _get_tf_examples_serving_signature(
            model, tf_transform_output
        ),
        "transform_features": _get_transform_features_signature(
            model, tf_transform_output
        ),
    }

    model.save(output_dir, save_format="tf", signatures=signatures)


# Function to build Keras model
def _build_keras_model(tf_transform_output: TFTransformOutput) -> tf.keras.Model:
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(_LABEL_KEY)

    inputs = {}
    for key, spec in feature_spec.items():
        if isinstance(spec, tf.io.VarLenFeature):
            inputs[key] = tf.keras.layers.Input(
                shape=[None], name=key, dtype=spec.dtype, sparse=True
            )
        elif isinstance(spec, tf.io.FixedLenFeature):
            inputs[key] = tf.keras.layers.Input(
                shape=spec.shape or [1], name=key, dtype=spec.dtype
            )
        else:
            raise ValueError(f"Spec type is not supported: {key}, {spec}")

    # Flatten all inputs and ensure they have the same shape before concatenation
    flattened_inputs = [
        tf.keras.layers.Flatten()(input_layer) for input_layer in inputs.values()
    ]
    x = tf.keras.layers.Concatenate()(flattened_inputs)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    output = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=inputs, outputs=output)


# Function to train the model
def run_fn(fn_args):
    tf_transform_output = TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(
        fn_args.train_files, tf_transform_output, batch_size=_BATCH_SIZE
    )
    eval_dataset = input_fn(
        fn_args.eval_files, tf_transform_output, batch_size=_BATCH_SIZE
    )

    model = _build_keras_model(tf_transform_output)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-1, decay_steps=1000, decay_rate=0.9
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="mean_squared_error",  # Using MSE for regression
        metrics=["mean_absolute_error"],
    )  # MAE as an additional metric

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq="batch"
    )
    logging.info("Training logs saved to: " + fn_args.model_run_dir)

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, early_stopping],
    )

    export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)


# Function to create Trainer component
def create_trainer(transform, schema_gen, module_file):
    return Trainer(
        module_file=module_file,
        custom_config={
            "ai_platform_training_args": {
                "project": GOOGLE_CLOUD_PROJECT,
                "region": GOOGLE_CLOUD_REGION,
                "job-dir": f"{GCS_BUCKET_NAME}/jobs",
            }
        },
        transformed_examples=transform.outputs["transformed_examples"],
        schema=schema_gen.outputs["schema"],
        transform_graph=transform.outputs["transform_graph"],
        train_args=trainer_pb2.TrainArgs(num_steps=500),
        eval_args=trainer_pb2.EvalArgs(num_steps=100),
    )
