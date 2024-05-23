# CO2 Emissions Prediction Using Machine Learning

Welcome to the repository where we explore the prediction of CO2 emissions from vehicles using machine learning techniques.

## Project Overview

This project aims to predict CO2 emissions based on various vehicle characteristics using a RandomForestRegressor model. The analysis is performed in a Jupyter Notebook that has been converted here for easy viewing.

## Dataset Description

The dataset includes several features such as engine size, cylinders, fuel consumption, etc., and was sourced from the Canadian government's public data on fuel consumption ratings.

## Data Preprocessing

```python
# Code for data preprocessing
import pandas as pd
df = pd.read_csv("FuelConsumption.csv")
df.drop(['MAKE','MODEL','VEHICLECLASS','TRANSMISSION','FUELTYPE'], axis=1, inplace=True)
