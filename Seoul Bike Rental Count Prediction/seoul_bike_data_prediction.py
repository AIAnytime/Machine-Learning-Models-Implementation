# -*- coding: utf-8 -*-
"""Seoul Bike Data Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-9fbZK0w0sEX07d3LtxrK-RrQaGHzyZg
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

"""# Data Source:
Link: https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand
"""

data = pd.read_csv("drive/My Drive/colab notebooks/SeoulBikeData.csv", encoding = 'unicode_escape')

data

"""# Narration:
- Features
- Target: Rented Bike Count

Needs to handle categorical columns.

# EDA
"""

data.info()

# check missing value
data.isna().sum()

data.corr()

corr = data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

print(data['Seasons'].value_counts())
print("-----------------------------")
print(data['Holiday'].unique())
print("------------------------------")
print(data['Functioning Day'].unique())

"""## Narrative:
- 3 categorical columns.
- 2 needs to be binary encoded and 1 needs to be One hot encoded.

# Data preprocessing

- Needs to use datetime object to extract features from the "Date" column.

- I will only use "Day and month" so this model can be serialized in any future predictions. Keeping year in the feature won't help the model by any means.

- will rename all the columns to lower case and in an organized way so it will be easy to use the model in deployment.

### I will use "Sklearn's Individual models and Pipeline" and "PyCaret" to perform the prediction that is mainly a "Regression Task".

# Using Pipeline
"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score

def preprocess_inputs(df):
  df = df.copy()

  #extract month and day from the Date column using datetime object
  df['Date'] = pd.to_datetime(df['Date'])
  df['month'] = df['Date'].apply(lambda x:x.month)
  df['day'] = df['Date'].apply(lambda x:x.day)

  #drop Date column
  df = df.drop("Date", axis=1)

  #rename the columns
  df.rename(columns={
      "Rented Bike Count": "rented_bike_count",
      "Hour": "hour",
      "Temperature(°C)": "temperature",
      "Humidity(%)": "humidity",
      "Wind speed (m/s)": "wind_speed",
      "Visibility (10m)": "visibility",
      "Dew point temperature(°C)": "dew_point_temperature",
      "Solar Radiation (MJ/m2)": "solar_radiation",
      "Rainfall(mm)": "rainfall",
      "Snowfall (cm)": "snowfall",
      "Seasons": "seasons",
      "Holiday": "holiday",
      "Functioning Day": "functioning_day"
  }, inplace=True)

  #performing binary encoding
  df['holiday'] = df['holiday'].apply(lambda x:1 if x == "Holiday" else 0)
  df['functioning_day'] = df['functioning_day'].apply(lambda x:1 if x == "Yes" else 0)

  #X and y
  X = df.drop("rented_bike_count", axis=1)
  y = df['rented_bike_count']

  #split
  X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, shuffle=True, random_state=21)

  return X_train, X_test, y_train, y_test

"""## Tree based models don't require scaled data so I am not using any "Standard Scaler" to scale the data.
i.e giving mean of 0 and variance of 1 to all the columns.
"""

X_train, X_test, y_train, y_test = preprocess_inputs(data)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

X_train

y_train

"""## Build the PIPELINE"""

nominal_transformer = Pipeline(steps=[
                          ('onehot', OneHotEncoder(sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
                                               ("nominal", nominal_transformer, ['seasons'])
], remainder = 'passthrough')

model = Pipeline(steps=[
                        ("preprocessor", preprocessor),
                        ("regressor", GradientBoostingRegressor())
])

#estimator = model.fit(X_train, np.ravel(y_train))
estimator = model.fit(X_train, y_train)

"""The numpy module of Python provides a function called numpy. ravel, which is used to change a 2-dimensional array or a multi-dimensional array into a contiguous flattened array. The returned array has the same data type as the source array or input array.

# Evaluation
"""

#score = estimator.score(X_test, y_test)
#print("Score:", np.round(score*100), "%")

y_true = np.array(y_test)
print(y_true)

y_pred = estimator.predict(X_test)
y_pred

#calculating the model's r-square
print("Model R^2 Score: {:.4f}".format(r2_score(y_true, y_pred)))

#calculating MSE & RMSE
print(np.mean((y_test - y_pred) ** 2))

rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print("RMSE is:", rmse)

"""## Using PyCaret"""

!pip install pycaret

import pycaret.regression as pyr

def data_preparation(df):
  df = df.copy()

  #extract month and day from the Date column using datetime object
  df['Date'] = pd.to_datetime(df['Date'])
  df['month'] = df['Date'].apply(lambda x:x.month)
  df['day'] = df['Date'].apply(lambda x:x.day)

  #drop Date column
  df = df.drop("Date", axis=1)

  #rename the columns
  df.rename(columns={
      "Rented Bike Count": "rented_bike_count",
      "Hour": "hour",
      "Temperature(°C)": "temperature",
      "Humidity(%)": "humidity",
      "Wind speed (m/s)": "wind_speed",
      "Visibility (10m)": "visibility",
      "Dew point temperature(°C)": "dew_point_temperature",
      "Solar Radiation (MJ/m2)": "solar_radiation",
      "Rainfall(mm)": "rainfall",
      "Snowfall (cm)": "snowfall",
      "Seasons": "seasons",
      "Holiday": "holiday",
      "Functioning Day": "functioning_day"
  }, inplace=True)

  #performing binary encoding
  df['holiday'] = df['holiday'].apply(lambda x:1 if x == "Holiday" else 0)
  df['functioning_day'] = df['functioning_day'].apply(lambda x:1 if x == "Yes" else 0)


  return df

X = data_preparation(data)

X

pyr.setup(
    data = X,
    target = 'rented_bike_count',
    train_size = 0.7,
    normalize = True
)

pyr.compare_models()

best_model = pyr.create_model("lightgbm")

pyr.evaluate_model(best_model)

"""## Hour and Temperature are the two most impactful features for the target and that's completely makes sense."""

pyr.save_model(best_model, "estimator_bike_rental_count")