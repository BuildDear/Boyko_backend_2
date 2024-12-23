import importlib
from data.datasets.gen_test_dataset import generate_10_data
import os

# Введення даних
date = input("Введіть дату (у форматі YYYY-MM-DD): ")

# Чи є івент
events = int(input("Чи є Events? (0 для ні, 1 для Holiday, 2 для Daily event, 3 для Promotion): ").strip())

if events == 0:
    test_data = generate_10_data(date, 0)
elif events == 1:
    test_data = generate_10_data(date, 1)
elif events == 2:
    test_data = generate_10_data(date, 2)
else:
    test_data = generate_10_data(date, 3)

dataset_path = os.path.abspath("data/datasets")
test_data.to_csv(f"{dataset_path}/10_rows.csv", index=False)

# Яку модель беремо
model_name = input("Введіть назву моделі (catboost, decision_tree, lightgbm, linearregression, xgboost): ").strip().lower()

# Маппінг для коректної капіталізації назв
model_name_mapping = {
    "catboost": "CatBoost",
    "decision_tree": "DecisionTree",
    "lightgbm": "LightGBM",
    "linearregression": "LinearRegression",
    "xgboost": "XGBoost"
}

# Коректне ім'я модуля
if model_name in model_name_mapping:
    model_class_name = model_name_mapping[model_name]
    module_path = f"edamonia_backend.logic.train.prediction.{model_class_name}"
    try:
        # Імпортуємо файл динамічно
        module = importlib.import_module(module_path)

        # Викликаємо функцію train та передаємо параметри
        print(f"Запуск моделі '{model_class_name}'...")

        current_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_path))
        dataset_path = os.path.join(project_root, "data", "datasets")

        module.train(events, dataset_path)  # Передаємо events і абсолютний шлях до Dataset

    except ModuleNotFoundError:
        print(f"Помилка: файл '{model_class_name}.py' не знайдено у директорії")
    except Exception as e:
        print(f"Помилка під час виконання моделі: {e}")
else:
    print("Помилка: введена некоректна назва моделі.")


'''створювати нові функції для тестового датасету щоб він відрізнявся від тренованого датасету
додати вивід метрик для 10_data'''