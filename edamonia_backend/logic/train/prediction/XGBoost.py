from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from xgboost import XGBRegressor
from edamonia_backend.logic.train.preprocess_data import preprocess_data, preprocess_test_data
import os

def train(events, dataset_path):
    print(f"Starting training process with events={events} and dataset_path='{dataset_path}'")

    if events == 0:
        file_path = os.path.join(dataset_path, "dataset.csv")
        test_path = os.path.join(dataset_path, "test_dataset.csv")
        print(f"Paths set for non-event dataset: file_path='{file_path}', test_path='{test_path}'")

        X_scaled, y = preprocess_data(file_path, 0)
        X_test, y_test = preprocess_data(test_path, 0)
        print("Data preprocessed for non-event dataset.")

        param_grid = {
            'n_estimators': [50],  # Кількість дерев
            'learning_rate': [0.15],  # Темп навчання
            'max_depth': [8]  # Глибина дерева
        }
        print("Parameter grid initialized for non-event dataset.")
    else:
        file_path = os.path.join(dataset_path, "dataset_event.csv")
        test_path = os.path.join(dataset_path, "test_dataset_event.csv")
        print(f"Paths set for event dataset: file_path='{file_path}', test_path='{test_path}'")

        X_scaled, y = preprocess_data(file_path, 1)
        X_test, y_test = preprocess_data(test_path, 1)
        print("Data preprocessed for event dataset.")

        param_grid = {
            'n_estimators': [50],  # Number of trees
            'learning_rate': [0.15],  # Learning rate
            'max_depth': [8]  # Tree depth
        }
        print("Parameter grid initialized for event dataset.")

    raw_test = pd.read_csv(test_path)
    print("Raw test dataset loaded:")
    print(raw_test.head())

    noise_level = 0.1  # Налаштуйте рівень шуму за потреби
    np.random.seed(42)  # Для відтворюваності
    noise = np.random.normal(0, noise_level, X_scaled.shape)
    X_train = X_scaled + noise
    print("Noise added to training data.")

    tscv = TimeSeriesSplit(n_splits=5)
    print("TimeSeriesSplit initialized with 5 splits.")

    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    print("XGBRegressor initialized.")

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1, verbose=1)
    print("GridSearchCV initialized with parameter grid:", param_grid)

    grid_search.fit(X_scaled, y)
    print("GridSearchCV fitting completed.")

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['mean_test_score'] = -results_df['mean_test_score']  # Convert MSE to positive for easier interpretation
    sorted_results = results_df.sort_values(by='mean_test_score').reset_index(drop=True)
    print("GridSearchCV results sorted.")

    selected_rows = sorted_results.iloc[[0, len(sorted_results) // 4, len(sorted_results) // 2, sorted_results['mean_test_score'].idxmin(), len(sorted_results) - 1]]
    print("Selected rows for analysis:")
    print(selected_rows)

    table = selected_rows[['param_n_estimators', 'param_learning_rate', 'param_max_depth', 'mean_test_score', 'std_test_score']].copy()
    table.columns = ['Number of trees', 'Learning rate', 'Tree depth', 'Mean MSE (cross-validation)', 'Std deviation MSE']
    print("Table for article created:")
    print(table)

    r2_scores = []
    for _, row in selected_rows.iterrows():
        model = XGBRegressor(
            n_estimators=row['param_n_estimators'],
            learning_rate=row['param_learning_rate'],
            max_depth=row['param_max_depth'],
            objective='reg:squarederror',
            random_state=42
        )
        r2 = cross_val_score(model, X_scaled, y, cv=tscv, scoring=make_scorer(r2_score)).mean()
        r2_scores.append(r2)
    print("R² scores calculated for selected models.")

    table['R² (cross-validation)'] = r2_scores
    print("R² scores added to the table:")
    print(table)

    results_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'prediction_results'))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Directory '{results_dir}' created.")

    table.to_csv(os.path.join(results_dir, 'XGBoost_results.csv'), index=False, encoding='utf-8-sig')
    print("Results table saved to CSV.")

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    best_r2 = cross_val_score(best_model, X_train, y, cv=tscv, scoring='r2').mean()
    print("Best model parameters and metrics determined:")
    print(f"Best parameters: {best_params}")
    print(f"Best MSE: {best_score}")
    print(f"Best R²: {best_r2}")

    best_model.fit(X_train, y)
    print("Best model fitted on training data.")

    y_test_pred = best_model.predict(X_test)
    print("Predictions made on test data.")

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print("Test metrics calculated:")
    print(f"MSE: {test_mse}")
    print(f"RMSE: {test_rmse}")
    print(f"MAE: {test_mae}")
    print(f"R²: {test_r2}")

    custom_test_file = f"{dataset_path}/10_rows.csv"
    custom_test_data = pd.read_csv(custom_test_file).copy()
    print(f"Custom test data loaded from '{custom_test_file}':")
    print(custom_test_data.head())

    X_custom, y_custom = preprocess_test_data(custom_test_file, events)
    print("Custom test data preprocessed.")

    custom_test_predictions = best_model.predict(X_custom)
    print("Predictions made on custom test data.")

    custom_test_data.loc[:, 'Predict_Quantity'] = custom_test_predictions
    custom_test_data = custom_test_data.drop('Purchase_Quantity', axis=1)
    custom_test_data.to_csv(os.path.join(results_dir, 'XGBoost_predict.csv'), index=False, encoding='utf-8-sig')
    print("Custom test predictions saved to CSV.")

    raw_test['Predicted_Purchase_Quantity'] = y_test_pred
    raw_test.to_csv(os.path.join(results_dir, 'XGBoost_test_predictions.csv'), index=False, encoding='utf-8-sig')
    print("Raw test predictions saved to CSV.")

    return {
        "model_name": "XGBoost",
        "parameters": {
            "n_estimators": best_params["n_estimators"],
            "learning_rate": best_params["learning_rate"],
            "max_depth": best_params["max_depth"]
        },
        "cv_metrics": {
            "mse": best_score
        },
        "test_metrics": {
            "mse": test_mse,
            "rmse": test_rmse,
            "mae": test_mae,
            "r2": test_r2
        }
    }