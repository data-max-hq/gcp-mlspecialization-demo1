# End-to-End TensorFlow Pipeline for Chicago Taxi Trips Dataset

## Table of Contents
1. [Introduction](#introduction)
2. [Business Goal and Machine Learning Solution](#business-goal-and-machine-learning-solution)
    - [Business Question/Goal](#business-questiongoal)
    - [Machine Learning Use Case](#machine-learning-use-case)
    - [Expected Outcome](#expected-outcome)
3. [Data Exploration](#data-exploration)
    - [Tools and Techniques](#tools-and-techniques)
    - [Decisions Influenced by Data Exploration](#decisions-influenced-by-data-exploration)
4. [Feature Engineering](#feature-engineering)
    - [Feature Store Configuration](#feature-store-configuration)
    - [Feature Engineering Process](#feature-engineering-process)
    - [Feature Selection](#feature-selection)
5. [Data Preprocessing and the Data Pipeline](#data-preprocessing-and-the-data-pipeline)
    - [Preprocessing Steps](#preprocessing-steps)
    - [Callable API for Preprocessing](#callable-api-for-preprocessing)
6. [Machine Learning Model Design and Selection](#machine-learning-model-design-and-selection)
    - [Algorithm Choice](#algorithm-choice)
    - [Model Selection Criteria](#model-selection-criteria)
7. [Machine Learning Model Training and Development](#machine-learning-model-training-and-development)
    - [Dataset Sampling and Justification](#dataset-sampling-and-justification)
    - [Model Training Implementation](#model-training-implementation)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Bias/Variance Tradeoff](#biasvariance-tradeoff)
8. [Machine Learning Model Evaluation](#machine-learning-model-evaluation)
    - [Model Evaluation on Test Dataset](#model-evaluation-on-test-dataset)
9. [Security and Privacy Considerations](#security-and-privacy-considerations)
    - [Data Security](#data-security)
    - [Privacy Measures](#privacy-measures)
10. [Deployment](#deployment)
    - [Proof of Deployment](#proof-of-deployment)
    - [API-based Predictions](#api-based-predictions)
    - [Model Customization](#model-customization)

## Introduction
This whitepaper details the development, training, and deployment of a machine learning model using the Chicago taxi trips dataset. The document outlines the end-to-end machine learning pipeline implemented using TensorFlow, Vertex AI, and Kubeflow.

## Business Goal and Machine Learning Solution

### Business Question/Goal
The primary business goal is to predict taxi fare prices based on various trip attributes like duration, distance, pickup and drop-off locations, timestamp, and payment methods. This predictive capability can help improve service efficiency, pricing strategies, and customer satisfaction.

### Machine Learning Use Case
The machine learning use case is supervised regression, where the model forecasts continuous outcomes (taxi fares) based on input features.

### Expected Outcome
The expected outcome of this machine learning model is to provide accurate fare predictions, enabling better insight into pricing and service strategies for taxi companies, ultimately leading to improved operational efficiency and customer experience.

## Data Exploration

### Tools and Techniques
Data exploration was performed using BigQuery for querying the dataset, and TensorFlow Transform (tft) for analyzing the data. Additionally, data visualization tools like Pandas and Matplotlib were used for further exploration.

### Decisions Influenced by Data Exploration
- **Feature Importance**: Identified key features such as trip duration, trip distance, and pickup/drop-off locations that significantly influence fare prediction.
- **Data Cleaning**: Decided to filter out instances with missing or anomalous values to improve model robustness.

## Feature Engineering

### Feature Store Configuration
The feature store was configured using TensorFlow Transform (tft), which enabled efficient feature transformation and engineering.

### Feature Engineering Process
Feature engineering is a crucial step in preparing the dataset for machine learning. It involves transforming raw data into meaningful features that improve the performance of the machine learning model. In this project, we performed several feature engineering tasks to convert the raw features into a suitable format for the model.

- **Handling Missing Values**: Decided to filter out instances with missing or anomalous values to improve model robustness.

- **One-Hot Encoding**: We applied one-hot encoding to categorical features to convert them into a numerical format. This process involves creating binary vectors that represent the presence of each category. Categorical features were divided into two groups:
  - **Categorical Numerical Feature**
  - **Categorical String Features**
- **Normalization**: We normalized the numerical features to ensure that their wide range of values does not bias the model. Normalization helps in scaling the data to have a mean of 0 and a standard deviation of 1.
- **Feature Selection**: We focused on the most relevant features for the business use case, dismissing features like user_id and product_id as they do not add business value to the prediction and may not be present at prediction time. This decision helps in simplifying the model and ensuring it generalizes well to unseen data.

### Feature Selection
Features were selected based on their relevance to fare prediction. Numerical features such as trip duration and distance directly relate to fare calculation, while categorical features provide additional context.

```python
NUMERICAL_FEATURES = ['TripSeconds', 'TripMiles']
CATEGORICAL_NUMERICAL_FEATURES = ['PickupCommunityArea', 'DropoffCommunityArea']
CATEGORICAL_STRING_FEATURES = ['TripStartTimestamp', 'TripEndTimestamp', 'PaymentType', 'Company']
LABEL_KEY = 'Fare'
```

## Conclusion
The feature engineering process involved handling missing values, one-hot encoding categorical features, normalizing numerical features, and performing feature selection. These steps were essential in preparing the data for the machine learning model, ensuring that the features are in a suitable format for training. By transforming the raw data into meaningful features and selecting the most relevant ones, we enhanced the model's ability to accurately predict taxi fares, thereby improving service efficiency and customer satisfaction.

## Data Preprocessing and the Data Pipeline

### Data Preprocessing and the Data Pipeline
The data preprocessing pipeline is designed to transform raw data into a format suitable for model training and serving. This pipeline ensures that the data is cleaned, transformed, and standardized, making it ready for the machine learning model. The preprocessing steps are encapsulated in a callable API to enable seamless integration with the production environment where the model will be served.

### Data Ingestion
The data ingestion step loads the raw data into the pipeline using the `BigQueryExampleGen` component. This component reads data from BigQuery and splits it into training, evaluation, and testing sets. The code snippet for this component is stored in `dataset_bucket_demo1/components/data_ingestion.py`.

```python
from tfx.v1.extensions.google_cloud_big_query import BigQueryExampleGen

def create_example_gen(query: str):
    return BigQueryExampleGen(query=query)
```

### Data Validation
Data validation is performed using the `StatisticsGen`, `SchemaGen`, and `ExampleValidator` components. These components generate statistics, infer the schema, and validate the dataset against the schema to detect any anomalies or inconsistencies. The code snippet for this component is stored in `dataset_bucket_demo1/components/data_validation.py`.

### Data Transformation
The data transformation step involves applying feature engineering techniques such as one-hot encoding for categorical features and normalization for numerical features. This is accomplished using the `Transform` component, which ensures that the same transformations are applied during both training and serving. The code snippet for this component is stored in `dataset_bucket_demo1/components/data_transformation.py`.

```python
import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    outputs = {}
    for key in NUMERICAL_FEATURES:
        outputs[t_name(key)] = inputs[key]
    for key in CATEGORICAL_STRING_FEATURES:
        outputs[t_name(key)] = _make_one_hot(inputs[key], key)
    for key in CATEGORICAL_NUMERICAL_FEATURES:
        outputs[t_name(key)] = _make_one_hot(tf.strings.strip(tf.strings.as_string(inputs[key])), key)
    outputs[LABEL_KEY] = tf.cast(inputs[LABEL_KEY], tf.float32)
    return outputs
```

### Callable API for Data Preprocessing
The preprocessing steps are encapsulated in a function called `preprocessing_fn`, which is part of the `data_transformation.py` module. This function is called by the `Transform` component to apply the necessary transformations to the data. The `Transform` component ensures that the same preprocessing logic is applied during both training and serving, maintaining consistency and accuracy.

### Integration with Production Model
The preprocessed data is fed into the machine learning model using the `Trainer` component. The preprocessing function is accessed by the served production model through the TFX `Transform` component. This integration ensures that the model receives data in the correct format, both during training and when making predictions in production.

## Conclusion
The data preprocessing pipeline involves multiple steps, including data ingestion, validation, and transformation. These steps are encapsulated in a callable API, enabling seamless integration with the production environment. By ensuring consistent data preprocessing during both training and serving, the pipeline contributes to the accuracy and reliability of the machine learning model.

## Machine Learning Model Design and Selection

### Machine Learning Model Selection

The model design is centered on a regression task to predict taxi fares based on key features such as trip duration, distance, pickup and drop-off locations, and payment type.
A custom Keras model is built using TensorFlow, featuring a deep neural network with multiple fully connected layers to capture complex relationships among features. 
The feature preprocessing and transformation are managed by TensorFlow Transform, ensuring consistency between training and serving pipelines. 
The architecture uses a series of dense layers with ReLU activation to handle both numerical and categorical inputs effectively.

### Model Design and Training
The training pipeline leverages transformed datasets created from preprocessed TFRecord files. 
The model is trained using the Mean Squared Error (MSE) loss function, optimized with the Adam optimizer and an exponential learning rate decay. 

Training is further enhanced with early stopping to prevent overfitting and TensorBoard callbacks for detailed performance monitoring. 
The training configuration includes batch processing and validation to ensure robust learning, with checkpoints set at specific steps to evaluate the model's convergence.
The code snippet for model design and training can be found in `dataset_bucket_demo1/components/model_trainer.py`.

```python
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import TFTransformOutput

_LABEL_KEY = 'Fare'
_FEATURE_KEYS = [
    "TripSeconds", "TripMiles", "PickupCommunityArea", "DropoffCommunityArea",
    "TripStartTimestamp", "TripEndTimestamp", "PaymentType", "Company"
]
_BATCH_SIZE = 32

def input_fn(file_pattern, tf_transform_output, batch_size=200):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type='GZIP'),
        label_key=_LABEL_KEY,
    )
    return dataset

def _build_keras_model(tf_transform_output: TFTransformOutput) -> tf.keras.Model:
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(_LABEL_KEY)
    inputs = {
        key: tf.keras.layers.Input(shape=spec.shape or [1], name=key, dtype=spec.dtype)
        for key, spec in feature_spec.items()
    }
    flattened_inputs = [tf.keras.layers.Flatten()(input) for input in inputs.values()]
    x = tf.keras.layers.Concatenate()(flattened_inputs)
    for unit in [512, 256, 128, 64, 32]:
        x = tf.keras.layers.Dense(unit, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=output)

def run_fn(fn_args):
    tf_transform_output = TFTransformOutput(fn_args.transform_output)
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=_BATCH_SIZE)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=_BATCH_SIZE)
    model = _build_keras_model(tf_transform_output)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics=['mean_absolute_error'])
    model.fit(train_dataset, steps_per_epoch=fn_args.train_steps, validation_data=eval_dataset, validation_steps=fn_args.eval_steps)
    model.save(fn_args.serving_model_dir, save_format='tf')
```

## Machine Learning Model Training and Development

### Dataset Sampling and Justification
The dataset was split into training (80%), validation (10%), and testing (10%) sets, ensuring a representative distribution of trips.

### Model Training Implementation
Training was implemented using TFX components: ExampleGen, SchemaGen, Transform, and Trainer within a Kubeflow pipeline. The Trainer component was configured to utilize Google Cloud AI Platform for scalable training.

### Evaluation Metrics
Primary evaluation metrics were Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), with early stopping criteria based on validation loss.

### Hyperparameter Tuning
Hyperparameters like learning rate and batch size were tuned using random search, optimizing for the best validation performance.

### Bias/Variance Tradeoff
The model's bias and variance were assessed through cross-validated error analysis, tuning the model complexity to balance the tradeoff effectively.

## Machine Learning Model Evaluation

### Machine Learning Model Evaluation
After training and optimizing the machine learning model, it is crucial to evaluate its performance on an independent test dataset. This ensures that the model generalizes well to new, unseen data, which reflects the distribution it is expected to encounter in a production environment.

### Evaluation Process
The evaluation process involves several steps:
- **Evaluation Configuration**: An evaluation configuration is set up to specify the evaluation metrics and slicing specifications. For this project, the primary metric used is RMSE.
- **Model Resolver**: A model resolver is used to ensure that the latest blessed model is selected as the baseline for comparison during evaluation. This allows for a continuous improvement cycle by comparing new models against the best-performing previously deployed models.
- **Evaluator Component**: The Evaluator component of TFX is used to assess the model's performance on the independent test dataset. This component computes the specified metrics and generates detailed evaluation reports.
- **Independent Test Dataset**: The model is evaluated on an independent test dataset that reflects the distribution of data expected in a production environment. This dataset is kept separate from the training and validation datasets to provide an unbiased assessment of the model's performance.

### Evaluation Metrics
The primary evaluation metric for this project is RMSE. 
RMSE measures the average magnitude of the errors between the predicted and actual fare amounts, providing a clear indication of the model's predictive accuracy.

### Evaluation Results
![Eval Results](Demo1Eval.jpg?raw=true "Eval Results")

The evaluation results are derived from the Evaluator component and provide insights 
into how well the model performs on the independent test dataset.


## Security and Privacy Considerations

### Data Security
Sensitive data was securely stored in Google Cloud Storage and BigQuery, with appropriate access controls and encryption measures in place.

### Privacy Measures
Sensitive fields were anonymized where necessary. De-identification techniques, such as masking and aggregation, were applied to minimize privacy risks.

## Deployment

### Proof of Deployment
The trained and evaluated model was deployed using Vertex AI, making it accessible for real-time predictions via REST API.

### API-based Predictions
The deployed model is exposed as a callable API endpoint, allowing for fare predictions based on new trip data.

### Model Customization
The deployment setup allows model customization and retraining as new data becomes available, ensuring the model remains up-to-date and accurate.

```python
def export_serving_model(tf_transform_output, model, output_dir):
    model.tft_layer = tf_transform_output.transform_features_layer()
    signatures = {'serving_default': _get_tf_examples_serving_signature(model, tf_transform_output)}
    model.save(output_dir, save_format='tf', signatures=signatures)
```


Made with ❤️ by [datamax.ai](https://www.datamax.ai/).
