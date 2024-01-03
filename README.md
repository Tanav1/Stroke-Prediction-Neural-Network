# Stroke Prediction Model using FastAPI

## Overview
This code demonstrates the development of a stroke prediction model using machine learning and the deployment of the model as a FastAPI web service. The model is trained on a dataset with various health-related features to predict the likelihood of a stroke occurrence.

## Code Description
1. **Data Preprocessing:**
   - Null values in the dataset are dropped using `df.dropna()`.
   - Label encoding is applied to categorical variables like 'gender,' 'ever_married,' 'work_type,' 'Residence_type,' and 'smoking_status' using `LabelEncoder`.

2. **Data Splitting and Resampling:**
   - The dataset is split into training and testing sets using `train_test_split`.
   - Random oversampling is performed on the training data using `RandomOverSampler` to address class imbalance.

3. **Model Training and Hyperparameter Tuning:**
   - Hyperparameter tuning is carried out using `GridSearchCV` to optimize an XGBoost model.
   - The best hyperparameters are used to train the XGBoost model.
   - Naive Bayes and another XGBoost model are also trained for comparison.

4. **Model Evaluation:**
   - The models are evaluated using metrics like Mean Squared Error (MSE) and accuracy score.
   - Confusion matrices are plotted using `seaborn` to visualize the model performance.

5. **Model Serialization:**
   - The best XGBoost model is serialized using `pickle` and saved as "model_tree.pkl."

6. **FastAPI Web Service:**
   - A FastAPI web service is created to provide predictions.
   - An example API endpoint "/predict" is provided to make predictions using the saved model.
   - The API accepts input data in JSON format and returns predictions.

## Running the Code
1. Install the required libraries mentioned in the code, such as `pandas`, `scikit-learn`, `xgboost`, `seaborn`, `fastapi`, and `colabcode`.

2. Execute the code in a Python environment.

3. Once the code is running, you can access the FastAPI endpoints for predictions.

4. Send POST requests to "/predict" with input data in JSON format to receive stroke predictions.

5. The trained XGBoost model should be stored in "model_tree.pkl" and loaded automatically when FastAPI service starts.

## Example Usage
You can use the following example JSON data to make predictions through the API endpoint:
```json
{
    "gender": 1,
    "age": 68,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": 1,
    "work_type": 2,
    "Residence_type": 1,
    "avg_glucose_level": 95,
    "bmi": 25,
    "smoking_status": 1
}
