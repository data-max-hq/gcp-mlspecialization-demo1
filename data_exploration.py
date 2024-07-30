import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
from google.cloud import bigquery

# Initialize BigQuery client
client = bigquery.Client()

# Define the query
query = """
SELECT
    CAST(trip_seconds AS FLOAT64) AS TripSeconds,
    CAST(trip_miles AS FLOAT64) AS TripMiles,
    CAST(pickup_community_area AS FLOAT64) AS PickupCommunityArea,
    CAST(dropoff_community_area AS FLOAT64) AS DropoffCommunityArea,
    CAST(trip_start_timestamp AS STRING) AS TripStartTimestamp,
    CAST(trip_end_timestamp AS STRING) AS TripEndTimestamp,
    CAST(payment_type AS STRING) AS PaymentType,
    CAST(company AS STRING) AS Company,
    CAST(fare AS FLOAT64) AS Fare
FROM
    `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE
    trip_seconds IS NOT NULL AND
    trip_miles IS NOT NULL AND
    pickup_community_area IS NOT NULL AND
    dropoff_community_area IS NOT NULL AND
    trip_start_timestamp IS NOT NULL AND
    trip_end_timestamp IS NOT NULL AND
    payment_type IS NOT NULL AND
    company IS NOT NULL AND
    fare IS NOT NULL AND
    trip_miles != 0 AND
    pickup_community_area != 0 AND
    dropoff_community_area != 0
LIMIT 10000
"""

# Execute the query
df = client.query(query).to_dataframe()

# Display a few rows of the dataframe
print(df.head())

# Basic statistics
print(df.describe())

# Missing values analysis
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Visualization: Distributions of numerical features
numerical_features = ['TripSeconds', 'TripMiles', 'Fare']
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], bins=50, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Visualization: Categorical features
categorical_features = ['PaymentType', 'Company']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df[feature], order=df[feature].value_counts().index)
    plt.title(f'Count of {feature}')
    plt.show()

# Scatter plot to explore relationships
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TripMiles', y='Fare', data=df)
plt.title('Fare vs TripMiles')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='TripSeconds', y='Fare', data=df)
plt.title('Fare vs TripSeconds')
plt.show()

# Box plot to detect outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x='Fare', data=df)
plt.title('Box plot of Fare')
plt.show()