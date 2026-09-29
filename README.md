# Taxi Trip Price Prediction

An end-to-end machine learning project for predicting taxi trip prices using **Simple Linear Regression (SLR)** and **Multiple Linear Regression (MLR)**.

The project follows a complete machine learning workflow, including exploratory data analysis, data preprocessing, baseline modeling, regression modeling, evaluation, residual analysis, and model persistence using a Scikit-learn pipeline.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Features](#features)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Baseline Model](#baseline-model)
- [Simple Linear Regression](#simple-linear-regression)
- [Multiple Linear Regression](#multiple-linear-regression)
- [Model Evaluation](#model-evaluation)
- [Error Analysis](#error-analysis)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [How to Run the Project](#how-to-run-the-project)
- [Technologies Used](#technologies-used)
- [Key Machine Learning Concepts](#key-machine-learning-concepts)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Learning Outcomes](#learning-outcomes)
- [Author](#author)

---



# Project Overview

This project predicts the price of a taxi trip based on characteristics such as:

- Trip distance
- Trip duration
- Passenger count
- Base fare
- Per-kilometre rate
- Per-minute rate
- Time of day
- Day of week
- Traffic conditions
- Weather conditions

Two Linear Regression approaches are explored:

### Simple Linear Regression

Uses only:

```text
Trip Distance → Trip Price
```



### Multiple Linear Regression

Uses multiple numerical and categorical features:

```text
Distance
Duration
Passenger Count
Base Fare
Per Km Rate
Per Minute Rate
Time of Day
Day of Week
Traffic Conditions
Weather
        ↓
    Trip Price
```

A simple mean-based regression model is also used as a **baseline** to determine whether the trained models provide meaningful predictive improvement.

---



# Problem Statement

Taxi trip prices depend on multiple factors, including distance, duration, pricing rates, traffic conditions, and other trip characteristics.

The objective of this project is to build regression models that can estimate the expected taxi trip price from these available features.

Formally:

```text
Given:
X = trip-related features

Predict:
y = Trip_Price
```

The project investigates two questions:

1. How well can taxi price be predicted using **trip distance alone**?
2. Does adding additional trip information improve the prediction compared with Simple Linear Regression and a mean-based baseline?

---



# Objectives

The main objectives of this project are:

- Understand and explore the taxi trip dataset.
- Identify data-quality issues.
- Analyze numerical and categorical variables.
- Handle missing values.
- Remove non-predictive identifier columns.
- Encode categorical variables.
- Scale numerical features where appropriate.
- Establish a simple regression baseline.
- Implement Simple Linear Regression.
- Implement Multiple Linear Regression.
- Evaluate models using standard regression metrics.
- Analyze residuals and prediction errors.
- Investigate Linear Regression assumptions.
- Build a reusable Scikit-learn preprocessing and modeling pipeline.
- Save the trained model for future predictions.

---



# Dataset

The dataset contains taxi trip information and the corresponding trip price.

### Dataset size

Approximately:

```text
1,000 observations
11 columns
```

The exact number of usable observations is reduced after removing rows where the target variable `Trip_Price` is missing.

### Target Variable

```text
Trip_Price
```

The target represents the price of the taxi trip.

---



# Features


| Feature                 | Type        | Description                           |
| ----------------------- | ----------- | ------------------------------------- |
| `id`                    | Identifier  | Unique identifier for the observation |
| `Trip_Distance_km`      | Numerical   | Distance travelled during the trip    |
| `Time_of_Day`           | Categorical | Time period of the trip               |
| `Day_of_Week`           | Categorical | Weekday or weekend                    |
| `Passenger_Count`       | Numerical   | Number of passengers                  |
| `Traffic_Conditions`    | Categorical | Traffic level during the trip         |
| `Weather`               | Categorical | Weather condition                     |
| `Base_Fare`             | Numerical   | Initial/base taxi fare                |
| `Per_Km_Rate`           | Numerical   | Price charged per kilometre           |
| `Per_Minute_Rate`       | Numerical   | Price charged per minute              |
| `Trip_Duration_Minutes` | Numerical   | Duration of the trip                  |
| `Trip_Price`            | Numerical   | **Target variable**                   |


---



# Machine Learning Workflow

The project follows the following workflow:

```text
Raw Dataset
     ↓
Exploratory Data Analysis
     ↓
Data Quality Checks
     ↓
Remove Missing Target
     ↓
Remove Identifier
     ↓
Train/Test Split
     ↓
Preprocessing
     ├── Numerical Imputation
     ├── Numerical Scaling
     ├── Categorical Imputation
     └── One-Hot Encoding
     ↓
Mean Baseline
     ↓
Simple Linear Regression
     ↓
Multiple Linear Regression
     ↓
Model Evaluation
     ↓
Error Analysis
     ↓
Final Model
     ↓
Save Model Pipeline
```

---



# Project Structure

```text
taxi-trip-price-prediction/
│
├── data/
│   └── taxi_trip_pricing.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_linear_regression.ipynb
│   └── 05_error_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── linear_regression_pipeline.pkl
│
├── reports/
│   └── figures/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---



# Exploratory Data Analysis

EDA was performed before model training to understand the structure and quality of the dataset.

The following aspects were investigated:

### Dataset structure

- Number of observations
- Number of features
- Data types
- Numerical and categorical variables



### Data quality

- Missing values
- Missing-value percentages
- Duplicate observations
- Unique categorical values



### Numerical analysis

Distributions of:

- Trip distance
- Passenger count
- Base fare
- Per-kilometre rate
- Per-minute rate
- Trip duration
- Trip price



### Categorical analysis

The following categorical variables were investigated:

- Time of day
- Day of week
- Traffic conditions
- Weather



### Relationship analysis

Relationships between numerical features and `Trip_Price` were investigated using scatter plots and correlation analysis.

One notable relationship is the strong positive association between:

```text
Trip_Distance_km
        ↓
Trip_Price
```

This relationship motivated the Simple Linear Regression experiment.

### Outlier analysis

Boxplots were used to identify potentially unusual observations.

Potential outliers were not automatically removed because an unusual taxi trip does not necessarily represent invalid data.

---



# Data Preprocessing

Preprocessing was performed after the train/test split to avoid data leakage.

## 1. Missing Target Values

Rows with missing `Trip_Price` values were removed.

The target cannot be reliably imputed for this supervised learning task because the actual price is the value we are trying to learn.

```python
df = df.dropna(subset=["Trip_Price"])
```

---



## 2. Remove Identifier

The `id` column was removed because it represents an identifier rather than a meaningful predictive feature.

```python
df = df.drop(columns=["id"])
```

---



## 3. Train/Test Split

The data was divided into:

```text
80% → Training
20% → Testing
```

using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The test set is kept separate until final model evaluation.

---



## 4. Numerical Features

Numerical features include:

```text
Trip_Distance_km
Passenger_Count
Base_Fare
Per_Km_Rate
Per_Minute_Rate
Trip_Duration_Minutes
```

Missing numerical values are handled using:

```python
SimpleImputer(strategy="median")
```

Numerical features are standardized using:

```python
StandardScaler()
```

---



## 5. Categorical Features

Categorical features include:

```text
Time_of_Day
Day_of_Week
Traffic_Conditions
Weather
```

Missing categorical values are handled using:

```python
SimpleImputer(strategy="most_frequent")
```

Categorical variables are converted into numerical representations using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The `handle_unknown="ignore"` option allows the pipeline to process previously unseen categories during inference without failing.

---



# Baseline Model

Before training Linear Regression, a simple baseline model was established.

The baseline predicts:

```text
Mean Trip_Price from the training set
```

for every test observation.

This means the baseline does not use any input features.

For example:

```text
Actual Trip 1 → Prediction = Mean Price
Actual Trip 2 → Prediction = Mean Price
Actual Trip 3 → Prediction = Mean Price
...
```

The baseline provides a reference point for determining whether Linear Regression is actually learning useful relationships from the features.

---



# Simple Linear Regression

The first machine learning model uses only:

```text
Trip_Distance_km
```

to predict:

```text
Trip_Price
```

The model follows:

$$
\hat{y} = \beta_0 + \beta_1x
$$

where:
$$
- \hat{y} = predicted trip price
$$
$$
- \beta_0 = intercept
$$
$$
- \beta_1 = coefficient
$$
- x = trip distance
The purpose of this experiment is to determine how much predictive information is contained in trip distance alone.

---



# Multiple Linear Regression

The second model uses multiple features to predict the taxi trip price.

The conceptual model is:

$$
\hat{y}
=
\beta_0
+
\beta_1X_1
+
\beta_2X_2
+
...
+
\beta_pX_p
$$

where the predictors include numerical and encoded categorical features.

The MLR model allows us to investigate whether adding additional information improves prediction compared with the distance-only SLR model.

---



# Model Evaluation

The models are evaluated using four standard regression metrics.

## Mean Absolute Error — MAE

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$$

MAE represents the average absolute prediction error.

Lower is better.

---



## Mean Squared Error — MSE

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y}_i)^2
$$

MSE penalizes larger errors more heavily because the errors are squared.

Lower is better.

---



## Root Mean Squared Error — RMSE

$$
RMSE =
\sqrt{MSE}
$$

RMSE is expressed in the same units as the target variable.

Lower is better.

---



## R² Score

R² measures the proportion of variation in the target explained by the model relative to a mean-prediction baseline.

A commonly used formulation is:

$$
R^2 =
1 -
\frac{\sum(y_i-\hat{y}_i)^2}
{\sum(y_i-\bar{y})^2}
$$

Higher values generally indicate better predictive fit.

---



# Results

The final models are compared using the same held-out test set.

Update the table below with the actual values obtained after running the notebooks.


| Model                      | MAE         | MSE         | RMSE        | R²          |
| -------------------------- | ----------- | ----------- | ----------- | ----------- |
| Mean Baseline              | `26.60`     | `2354.96`   | `48.52`     | `-0.007`    |
| Simple Linear Regression   | `16.92`     | `572.47`    | `23.92`     | `0.75`      |
| Multiple Linear Regression | `9.93`      | `289.98`    | `17.02`     | `0.87`      |




### Interpretation

The baseline establishes the minimum reference performance.

Simple Linear Regression tests whether trip distance alone provides useful predictive information.

Multiple Linear Regression tests whether incorporating additional trip characteristics improves predictive performance.

The final model should be evaluated using multiple metrics rather than relying on R² alone.

---



# Error Analysis

Error analysis was performed on the final Multiple Linear Regression model.

For each test observation, the following values were calculated:

```text
Actual
Predicted
Residual
Absolute Error
Percentage Error
```

The residual is defined as:

$$
e_i = y_i-\hat{y}_i
$$

where:

- Positive residual → model underpredicted
- Negative residual → model overpredicted

---



## Diagnostic Analysis

The following diagnostics were performed:

### Actual vs Predicted

Used to determine how closely predictions follow the actual values.

### Residual Distribution

Used to inspect whether residuals are approximately centered around zero and whether unusual patterns exist.

### Residuals vs Predictions

Used to investigate:

- Non-linearity
- Systematic prediction patterns
- Heteroscedasticity



### Residuals vs Features

Residuals were investigated against important variables such as:

- Trip distance
- Trip duration



### Largest Prediction Errors

The observations with the largest absolute errors were inspected to identify potentially difficult cases or data-quality issues.

### Error by Distance Group

Prediction errors were compared across different distance ranges to determine whether model performance changes for short and long trips.

---



# Linear Regression Assumptions

The project also considers important Linear Regression assumptions.

## Linearity

The relationship between predictors and the target should be adequately represented by a linear model.

Investigated using:

- Scatter plots
- Residual plots



## Independence

Observations and errors should be reasonably independent.

This assumption depends largely on how the dataset was collected.

## Homoscedasticity

Residual variance should remain approximately constant across predicted values.

Investigated using:

```text
Residuals vs Predicted Values
```



## Residual Distribution

The residual distribution was inspected visually.

Normality of residuals is primarily important for classical statistical inference rather than being an absolute requirement for making useful predictions.

## Multicollinearity

Correlations between numerical predictors were inspected to identify potentially redundant information.

Potential multicollinearity is particularly relevant when predictors describe related aspects of the same trip.

---



# Model Pipeline

The final model uses a Scikit-learn `Pipeline` and `ColumnTransformer`.

Conceptually:

```text
Raw Input
    │
    ▼
ColumnTransformer
    │
    ├── Numerical Features
    │      ├── Median Imputation
    │      └── StandardScaler
    │
    └── Categorical Features
           ├── Most-Frequent Imputation
           └── One-Hot Encoding
    │
    ▼
Linear Regression
    │
    ▼
Prediction
```

This approach prevents preprocessing logic from becoming disconnected from the trained model.

The entire pipeline can be saved and later used for inference.

---



# Model Persistence

The final model pipeline is saved using `joblib`.

```python
import joblib

joblib.dump(
    mlr_pipeline,
    "../models/linear_regression_pipeline.pkl"
)
```

The saved artifact contains:

```text
Preprocessing
+
Linear Regression Model
```

This means new raw observations can be passed directly into the pipeline without manually repeating imputation, encoding, and scaling.

---



# Example Prediction

After loading the trained pipeline:

```python
import joblib
import pandas as pd

model = joblib.load(
    "models/linear_regression_pipeline.pkl"
)

new_trip = pd.DataFrame({
    "Trip_Distance_km": [15],
    "Time_of_Day": ["Evening"],
    "Day_of_Week": ["Weekday"],
    "Passenger_Count": [2],
    "Traffic_Conditions": ["Medium"],
    "Weather": ["Clear"],
    "Base_Fare": [3.5],
    "Per_Km_Rate": [1.2],
    "Per_Minute_Rate": [0.3],
    "Trip_Duration_Minutes": [40]
})

prediction = model.predict(new_trip)

print("Predicted Trip Price:", prediction[0])
```

---



# Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Joblib**
- **Jupyter Notebook**

---



# Key Machine Learning Concepts

This project demonstrates practical implementation of:

### Data Analysis

- Data inspection
- Missing-value analysis
- Duplicate detection
- Numerical analysis
- Categorical analysis
- Correlation analysis
- Outlier investigation



### Data Preprocessing

- Train/test splitting
- Missing-value imputation
- One-hot encoding
- Feature scaling
- ColumnTransformer
- Pipeline



### Supervised Learning

- Regression
- Simple Linear Regression
- Multiple Linear Regression



### Model Evaluation

- MAE
- MSE
- RMSE
- R²



### Model Diagnostics

- Residual analysis
- Actual vs predicted analysis
- Error analysis
- Heteroscedasticity investigation
- Linearity investigation
- Multicollinearity investigation



### ML Engineering

- Reproducible train/test split
- Prevention of data leakage
- Reusable preprocessing pipeline
- Model serialization
- Separate notebooks for different stages

---



# Limitations

This project has several limitations.

### Dataset size

The dataset contains approximately 1,000 observations, which is relatively small for a production-grade prediction system.

### Linear model

Linear Regression assumes that the relationships can be adequately represented using a linear combination of the input features.

Real-world taxi pricing may contain nonlinear relationships.

### Feature limitations

The dataset does not necessarily include every factor that could influence taxi pricing, such as:

- Pickup location
- Drop-off location
- Geographic demand
- Surge pricing
- Holiday effects
- Special events
- Driver availability
- Real-time supply and demand



### Dataset realism

The dataset is suitable for learning and experimentation, but additional validation would be required before using such a model in a real-world pricing system.

---



# Future Improvements

The project can be extended in several directions.

## Feature Engineering

Potential features include:

- Distance/time interaction
- Fare-per-distance features
- Fare-per-minute features
- Peak-hour indicators
- Weekend indicators
- Nonlinear transformations



## Regularized Linear Models

Compare Linear Regression with:

- Ridge Regression
- Lasso Regression
- Elastic Net



## Tree-Based Models

Experiment with:

- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost



## Model Selection

Use:

- Cross-validation
- GridSearchCV
- RandomizedSearchCV

for systematic model comparison and hyperparameter tuning.

## Deployment

The trained model could eventually be exposed through:

- FastAPI
- Flask
- Streamlit

and deployed as an inference service.

## Monitoring

A production system could additionally monitor:

- Prediction distributions
- Input-data drift
- Model performance
- Missing-value rates
- Feature drift
- Prediction latency

---



# Learning Outcomes

Through this project, I practiced the complete workflow of a supervised machine learning regression problem:

```text
Problem Definition
       ↓
Data Understanding
       ↓
EDA
       ↓
Data Cleaning
       ↓
Train/Test Split
       ↓
Preprocessing
       ↓
Baseline
       ↓
Simple Linear Regression
       ↓
Multiple Linear Regression
       ↓
Evaluation
       ↓
Error Analysis
       ↓
Model Pipeline
       ↓
Model Persistence
```

The project also helped develop an understanding of why preprocessing must be performed correctly to avoid data leakage and why a model should always be compared against a meaningful baseline.

---



# How to Run the Project



## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```



## 2. Navigate to the project

```bash
cd taxi-trip-price-prediction
```



## 3. Create a virtual environment



### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```



### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```



## 4. Install dependencies

```bash
pip install -r requirements.txt
```



## 5. Start Jupyter

```bash
jupyter notebook
```



## 6. Run the notebooks in order

```text
01_eda.ipynb
       ↓
02_preprocessing.ipynb
       ↓
03_baseline_model.ipynb
       ↓
04_linear_regression.ipynb
       ↓
05_error_analysis.ipynb
```

---



# Reproducibility

The project uses:

```python
random_state=42
```

for the train/test split.

This ensures that the same observations are assigned to the training and testing sets when the workflow is rerun under the same conditions.

---

