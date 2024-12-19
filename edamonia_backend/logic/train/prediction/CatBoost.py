from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import pandas as pd
import numpy as np
from edamonia_backend.logic.train.preprocess_data import preprocess_data, preprocess_test_data
import os


def train(events, dataset_path):
    if events != 0:
        # Step 1: Load the dataset
        file_path = os.path.join(dataset_path, "data_without_events.csv")

        X_scaled, y, kf = preprocess_data(file_path, 1)

        # Step 2: Define parameter grid for GridSearchCV
        #param_grid = {
        #    'iterations': [350, 400, 450],  # Кількість ітерацій
        #    'learning_rate': [0.04, 0.05],  # Темп навчання
        #    'depth': [6, 7]  # Глибина дерева
        #}
        param_grid = {
            'iterations': [400],  # Кількість ітерацій
            'learning_rate': [0.04],  # Темп навчання
            'depth': [6]  # Глибина дерева
        }
    else:
        file_path = os.path.join(dataset_path, "data_without_events.csv")

        # Preprocess data
        X_scaled, y, kf = preprocess_data(file_path, 0)
        param_grid = {
            'iterations': [400],  # Кількість ітерацій
            'learning_rate': [0.04],  # Темп навчання
            'depth': [7]  # Глибина дерева
        }
        # Step 2: Define parameter grid for GridSearchCV
        # param_grid = {
        #     'iterations': [300, 400, 500],  # Кількість ітерацій
        #     'learning_rate': [0.03, 0.04, 0.05],  # Темп навчання
        #     'depth': [5, 6, 7]  # Глибина дерева
        # }

    catboost_model = CatBoostRegressor(verbose=0, random_state=42)

    # Step 4: Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=catboost_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',  # Оптимізація за MSE
        cv=kf,
        n_jobs=-1,
        verbose=1
    )

    # Step 5: Fit GridSearchCV
    grid_search.fit(X_scaled, y)

    # Step 6: Convert results to DataFrame and sort by mean_test_score
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['mean_test_score'] = -results_df['mean_test_score']  # Convert MSE to positive for easier interpretation
    sorted_results = results_df.sort_values(by='mean_test_score').reset_index(drop=True)

    # Step 7: Select rows for the table
    selected_rows = sorted_results.head(5)  # Top 5 results

    # Step 8: Create table for the article
    table = selected_rows[
        ['param_iterations', 'param_learning_rate', 'param_depth', 'mean_test_score', 'std_test_score']
    ].copy()
    table.columns = [
        'Кількість ітерацій', 'Темп навчання', 'Глибина дерева',
        'Середнє MSE (крос-валідація)', 'Стандартне відхилення MSE'
    ]

    # Step 9: Compute R² for each selected model
    r2_scores = []
    for _, row in selected_rows.iterrows():
        model = CatBoostRegressor(
            iterations=row['param_iterations'],
            learning_rate=row['param_learning_rate'],
            depth=row['param_depth'],
            verbose=0,
            random_state=42
        )
        r2 = cross_val_score(model, X_scaled, y, cv=kf, scoring=make_scorer(r2_score)).mean()
        r2_scores.append(r2)

    # Step 10: Add R² column to the table
    table['R² (крос-валідація)'] = r2_scores

    # Step 11: Save table to CSV
    table.to_csv('edamonia_backend/logic/train/prediction_results/CatBoost_results.csv', index=False, encoding='utf-8-sig')

    # Step 12: Display table
    # print("\nТаблиця результатів:")
    # print(table)

    # Step 13: Train and evaluate the best model
    best_model = grid_search.best_estimator_

    # Step 7: Get the best model's parameters
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    # Step 8: Print the best model parameters
    print("\nПараметри найкращої моделі CatBoost:")
    print(f"Кількість ітерацій: {best_params['iterations']}")
    print(f"Темп навчання: {best_params['learning_rate']}")
    print(f"Глибина дерева: {best_params['depth']}")
    print(f"Середнє MSE (крос-валідація): {best_score:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    best_model.fit(X_train, y_train)

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
    custom_test_data = pd.read_csv(custom_test_file)

    X_custom, y_custom = preprocess_test_data(custom_test_file, events)

    # Step 15: Make predictions on the custom test table
    custom_test_predictions = best_model.predict(X_custom)

    # Step 16: Add predictions to the custom test table
    custom_test_data['Прогноз'] = custom_test_predictions
    # Step 17: Save the updated table with predictions
    custom_test_data.to_csv('edamonia_backend/logic/train/prediction_results/CatBoost_predict.csv', index=False, encoding='utf-8-sig')

    # Step 18: Display the updated custom test table
    # print("\nТаблиця з прогнозами:")
    # print(custom_test_data.head())

    return {
        "model_name": "CatBoost",
        "parameters": {
            "iterations": best_params["iterations"],
            "learning_rate": best_params["learning_rate"],
            "depth": best_params["depth"]
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
