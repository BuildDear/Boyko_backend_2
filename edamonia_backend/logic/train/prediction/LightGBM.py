from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import numpy as np
from edamonia_backend.logic.train.preprocess_data import preprocess_data, preprocess_test_data
import pandas as pd
import os
import warnings
import sys

# Придушення FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Придушення будь-яких додаткових попереджень з joblib і sklearn
if not sys.warnoptions:
    os.environ["PYTHONWARNINGS"] = "ignore"

def train(events, dataset_path):

    if events == 0:
        # Step 1: Load the dataset
        file_path = os.path.join(dataset_path, "data_without_events.csv")

        # Preprocessing data
        X_scaled, y, kf = preprocess_data(file_path, 1)

        param_grid = {
            'n_estimators': [50, 100, 200],  # Кількість дерев
            'learning_rate': [0.05, 0.1, 0.2],  # Темп навчання
            'max_depth': [3, 5, 7],  # Глибина дерева
            'num_leaves': [15, 31, 50]  # Кількість листків
        }
    else:
        file_path = os.path.join(dataset_path, "data_with_events.csv")

        # Preprocessing data
        X_scaled, y, kf = preprocess_data(file_path, 1)

        param_grid = {
            'n_estimators': [50, 100, 200],  # Кількість дерев
            'learning_rate': [0.05, 0.1, 0.2],  # Темп навчання
            'max_depth': [3, 5, 7],  # Глибина дерева
            'num_leaves': [15, 31, 50]  # Кількість листків
        }

    # Step 3: Define LightGBM model
    lgbm_model = LGBMRegressor(objective='regression', random_state=42, verbose=-1)

    # Step 4: Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=lgbm_model,
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
    selected_rows = sorted_results.iloc[
        [0, len(sorted_results) // 4, len(sorted_results) // 2, sorted_results['mean_test_score'].idxmin(), len(sorted_results) - 1]
    ]

    # Step 8: Create table for the article
    table = selected_rows[
        ['param_n_estimators', 'param_learning_rate', 'param_max_depth', 'param_num_leaves', 'mean_test_score', 'std_test_score']
    ].copy()
    table.columns = [
        'Кількість дерев', 'Темп навчання', 'Глибина дерева', 'Кількість листків',
        'Середнє MSE (крос-валідація)', 'Стандартне відхилення MSE'
    ]

    # Step 9: Compute R² for each selected model
    r2_scores = []
    for _, row in selected_rows.iterrows():
        model = LGBMRegressor(
            n_estimators=row['param_n_estimators'],
            learning_rate=row['param_learning_rate'],
            max_depth=row['param_max_depth'],
            num_leaves=row['param_num_leaves'],
            objective='regression',
            random_state=42,
            verbose=-1
        )
        r2 = cross_val_score(model, X_scaled, y, cv=kf, scoring=make_scorer(r2_score)).mean()
        r2_scores.append(r2)

    # Step 10: Add R² column to the table
    table['R² (крос-валідація)'] = r2_scores

    # Step 11: Save table to CSV
    table.to_csv('edamonia_backend/logic/train/prediction_results/LightGBM_results.csv', index=False, encoding='utf-8-sig')

    # Step 11: Make predictions using the best model from GridSearchCV
    best_model = grid_search.best_estimator_
    # Step 7: Get the best model's parameters
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_

    # Step 8: Print the best model parameters
    print("\nПараметри найкращої моделі LightGBM:")
    print(f"Кількість ітерацій: {best_params['n_estimators']}")
    print(f"Темп навчання: {best_params['learning_rate']}")
    print(f"Максимальна глибина дерева: {best_params['max_depth']}")
    print(f"Кількість листків: {best_params['num_leaves']}")
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
    custom_test_data.to_csv('edamonia_backend/logic/train/prediction_results/LightGBM_predict.csv', index=False, encoding='utf-8-sig')

    # Step 18: Display the updated custom test table
    print("\nТаблиця з прогнозами:")
    print(custom_test_data.head())
