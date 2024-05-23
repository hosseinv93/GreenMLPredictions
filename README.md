# CO2 Emission Prediction

This repository contains a Jupyter Notebook that demonstrates the prediction of CO2 emissions based on vehicle characteristics using various machine learning techniques. The project utilizes a RandomForestRegressor and is fine-tuned through GridSearchCV to achieve the best prediction results. Additionally, SHAP values are computed to interpret the model's predictions.

## Project Objective

Welcome to the repository where we explore the prediction of CO2 emissions from vehicles using machine learning techniques.
This project aims to predict CO2 emissions based on various vehicle characteristics using a RandomForestRegressor model. The analysis is performed in a Jupyter Notebook that has been converted here for easy viewing.


## Dataset

The dataset used in this project is the `FuelConsumption.csv`, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. You can download the dataset directly from [IBM Cloud](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv).

## Features

The dataset includes features like:
- `MAKE`: Manufacturer of the vehicle
- `MODEL`: Model of the vehicle
- `VEHICLE CLASS`: Vehicle class
- `ENGINE SIZE`: Engine size in liters
- `CYLINDERS`: Number of cylinders
- `TRANSMISSION`: Type of transmission
- `FUEL TYPE`: Type of fuel
- `FUEL CONSUMPTION in CITY(L/100 km)`: Fuel consumption in city
- `FUEL CONSUMPTION in HWY (L/100 km)`: Fuel consumption on highway
- `FUEL CONSUMPTION COMB (L/100 km)`: Combined fuel consumption
- `CO2 EMISSIONS`: CO2 emissions in grams per km

## Installation

To run this notebook, you'll need to install the necessary Python libraries. You can install the dependencies using the following command:

```bash
pip install numpy pandas matplotlib scikit-learn shap
```

### Downloading the Dataset

To download the dataset directly into the directory where you plan to run the notebook, use the following command:

```bash
wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv
```


## Data Preprocessing
The dataset is preprocessed to remove irrelevant columns and shuffle the data for randomness. The preprocessed data is then split into training and testing sets.

```python
import pandas as pd

df = pd.read_csv("FuelConsumption.csv")
df_new = df.drop(['MAKE','MODEL','VEHICLECLASS','TRANSMISSION','FUELTYPE'], axis=1)
data = df_new.sample(frac=1, random_state=42)
train_set, test_set = train_test_split(data, test_size=0.15, random_state=42)
```
