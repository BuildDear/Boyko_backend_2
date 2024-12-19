import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from edamonia_backend.logic.train.preprocess_data import preprocess_data, preprocess_test_data
import os


def train(events, dataset_path):
    if events == 0:
        # Step 1: Load the dataset
        file_path = os.path.join(dataset_path, "data_without_events.csv")

        # Preprocessing data
        X_scaled, y, kf = preprocess_data(file_path, 1)
    else:
        file_path = os.path.join(dataset_path, "data_with_events.csv")

        # Preprocessing data
        X_scaled, y, kf = preprocess_data(file_path, 1)

    # Step 2: Define the Linear Regression model
    lin_reg_model = LinearRegression()


    # Step 7: Train the model on the entire training set for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    lin_reg_model.fit(X_train, y_train)

    # Step 8: Make predictions and evaluate on the test set
    y_test_pred = lin_reg_model.predict(X_test)

    # Step 9: Calculate evaluation metrics for the test set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Step 10: Save test set results to a DataFrame
    test_results_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
        'Value': [test_mse, test_rmse, test_mae, test_r2]
    })

    # Step 11: Save test set results to CSV
    test_results_df.to_csv('edamonia_backend/logic/train/prediction_results/LinearRegression_results.csv', index=False, encoding='utf-8-sig')

    print("\nTest Set Metrics:")
    print(test_results_df)

    # Step 12: Load the custom test file
    custom_test_file = f"{dataset_path}/10_rows.csv"
    custom_test_data = pd.read_csv(custom_test_file)

    # Step 13: Preprocess the custom test data
    X_custom, y_custom = preprocess_test_data(custom_test_file, events)

    # Step 14: Make predictions on the custom test table
    custom_test_predictions = lin_reg_model.predict(X_custom)

    # Step 15: Add predictions to the custom test table
    custom_test_data.loc[:, 'Прогноз'] = custom_test_predictions

    # Step 16: Save the updated table with predictions
    custom_test_data.to_csv('edamonia_backend/logic/train/prediction_results/LinearRegression_predict.csv', index=False, encoding='utf-8-sig')

    # # Step 17: Display the updated custom test table
    # print("\nТаблиця з прогнозами:")
    # print(custom_test_data.head())

    return {
        "model_name": "LinearRegression",
        "parameters": None,  # Немає параметрів
        "cv_metrics": None,  # Немає крос-валідації
        "test_metrics": {
            "mse": test_mse,
            "rmse": test_rmse,
            "mae": test_mae,
            "r2": test_r2
        }
    }