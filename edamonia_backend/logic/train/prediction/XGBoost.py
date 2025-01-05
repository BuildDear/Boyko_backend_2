from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from xgboost import XGBRegressor
from edamonia_backend.logic.train.preprocess_data import preprocess_data, preprocess_test_data
import os

def train(events, dataset_path):

    if events == 0:
        file_path = os.path.join(dataset_path, "dataset.csv")
        test_path = os.path.join(dataset_path, "test_dataset.csv")

        X_scaled, y = preprocess_data(file_path, 0)
        X_test, y_test = preprocess_data(test_path, 0)

        param_grid = {
            'n_estimators': [50],  # Кількість дерев
            'learning_rate': [0.15],  # Темп навчання
            'max_depth': [8]  # Глибина дерева
        }
    else:
        file_path = os.path.join(dataset_path, "dataset_event.csv")
        test_path = os.path.join(dataset_path, "test_dataset_event.csv")

        X_scaled, y = preprocess_data(file_path, 1)
        X_test, y_test = preprocess_data(test_path, 1)

        param_grid = {
            'n_estimators': [50],  # Number of trees
            'learning_rate': [0.15],  # Learning rate
            'max_depth': [8]  # Tree depth
        }
    raw_test = pd.read_csv(test_path)

    noise_level = 0.1  # Налаштуйте рівень шуму за потреби
    np.random.seed(42)  # Для відтворюваності
    noise = np.random.normal(0, noise_level, X_scaled.shape)
    X_train = X_scaled + noise

    tscv = TimeSeriesSplit(n_splits=5)
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    # print(hasattr(xgb_model, "__sklearn_tags__"))  # Це має повернути False
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1, verbose=1)

    # Step 11: Fit the GridSearchCV
    grid_search.fit(X_scaled, y)

    # Convert results to DataFrame and sort by mean_test_score
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['mean_test_score'] = -results_df['mean_test_score']  # Convert MSE to positive for easier interpretation
    sorted_results = results_df.sort_values(by='mean_test_score').reset_index(drop=True)

    # Select rows for table: Base model, Intermediate models, Best model, Slightly worse than best
    selected_rows = sorted_results.iloc[[0, len(sorted_results) // 4, len(sorted_results) // 2, sorted_results['mean_test_score'].idxmin(), len(sorted_results) - 1]]

    # Create table for the article
    table = selected_rows[['param_n_estimators', 'param_learning_rate', 'param_max_depth', 'mean_test_score', 'std_test_score']].copy()
    table.columns = ['Number of trees', 'Learning rate', 'Tree depth', 'Mean MSE (cross-validation)', 'Std deviation MSE']

    # Compute R² for each selected model
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

    table['R² (cross-validation)'] = r2_scores

    # Save the results table to CSV
    table.to_csv('edamonia_backend/logic/train/prediction_results/XGBoost_results.csv', index=False, encoding='utf-8-sig')

    # Step 11: Make predictions using the best model from GridSearchCV
    best_model = grid_search.best_estimator_
    # Step 7: Get the best model's parameters
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    best_r2 = cross_val_score(best_model, X_train, y, cv=tscv, scoring='r2').mean()

    # Step 8: Print the best model parameters
    print("\nПараметри найкращої моделі XGBoost:")
    print(f"Кількість ітерацій: {best_params['n_estimators']}")
    print(f"Темп навчання: {best_params['learning_rate']}")
    print(f"Максимальна глибина дерева: {best_params['max_depth']}")
    print(f"Середнє MSE (крос-валідація): {best_score:.4f}")
    print(f"R² найкращої моделі: {best_r2:.4f}")

    best_model.fit(X_train, y)

    y_test_pred = best_model.predict(X_test)

    # Step 14: Calculate evaluation metrics for the test set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nTest Set Metrics:")
    print(f"Mean Squared Error (MSE): {test_mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
    print(f"R-squared (R²): {test_r2:.4f}")

    custom_test_file = f"{dataset_path}/10_rows.csv"
    custom_test_data = pd.read_csv(custom_test_file).copy()

    X_custom, y_custom = preprocess_test_data(custom_test_file, events)

    custom_test_predictions = best_model.predict(X_custom)

    custom_test_data.loc[:, 'Predict_Quantity'] = custom_test_predictions
    custom_test_data = custom_test_data.drop('Purchase_Quantity', axis=1)
    custom_test_data.to_csv('edamonia_backend/logic/train/prediction_results/XGBoost_predict.csv', index=False, encoding='utf-8-sig')

    raw_test['Predicted_Purchase_Quantity'] = y_test_pred

    # Збереження нової таблиці до CSV
    raw_test.to_csv('edamonia_backend/logic/train/prediction_results/XGBoost_test_predictions.csv', index=False, encoding='utf-8-sig')

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
