import pandas as pd
from data.datasets.additional_functions import *
from datetime import datetime


# Parameters for data generation
products = ['Milk', 'Eggs', 'Chicken', 'Tomatoes', 'Apples', 'Salmon', 'Cheese', 'Lettuce', 'Pork', 'Potatoes']
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weather_conditions = ['Sunny', 'Rainy', 'Snowy', 'Cloudy', 'Stormy', 'Hot', 'Cold']

# Об'єднана функція для генерації тестових даних
def generate_test_data(date, is_event=None):
    """
    Генерує тестові дані на основі дати та події (якщо є).

    Args:
        date (str | datetime): Дата у форматі 'YYYY-MM-DD' або об'єкт datetime.
        is_event (str, optional): Назва події (наприклад, свято). За замовчуванням None.

    Returns:
        pd.DataFrame: Згенеровані дані у вигляді DataFrame.
    """
    data = []

    # Конвертація date у datetime, якщо це рядок
    if isinstance(date, str):
        current_date = datetime.strptime(date, "%Y-%m-%d")
    else:
        current_date = date

    event = None
    if is_event == 0:
        event = 'None'
    elif is_event == 1:
        event = 'Holiday'
    elif is_event == 2:
        event = 'Daily event'
    elif is_event == 3:
        event = 'Promotion'

    # Визначаємо сезон і погоду
    season = determine_season(current_date)
    weather = random.choice(seasonal_weather[season])
    num_customers = generate_num_customers(current_date, season, weather)
    sales = get_average_check(num_customers)

    # Генерація даних для кожного продукту
    for i in range(len(products)):
        product = products[i]
        year = current_date.year
        day_of_week = days_of_week[current_date.weekday()]
        stocks = get_stock(num_customers, event)
        days_until_next_purchase = next_purchase(stocks, product)
        category = get_category(product)
        quantity = determine_quantity(num_customers, stocks, season, product, event)
        unit_price = get_price(season, product, year)
        shelf_life = get_shelf_life(product)

        # Додаємо дані до списку
        data.append([
            current_date, day_of_week, season, weather, product, category, unit_price,
            num_customers, sales, stocks, shelf_life, days_until_next_purchase, event, quantity
        ])

    # Створення DataFrame
    columns = ['Date', 'Day_of_Week', 'Season', 'Weather', 'Product', 'Category', 'Unit_Price',
               'Num_Customers', 'Sales', 'Stocks', 'Shelf_Life', 'Days_Until_Next_Purchase',
               'Event', 'Purchase_Quantity']
    df = pd.DataFrame(data, columns=columns)
    print("Test dataset generated successfully")
    return df


# # Generate synthetic data
# synthetic_data = generate_test_data_events(2022-12-12)
#
# # To display the first 5 rows of the DataFrame
# print(synthetic_data.head())
#
# # Or, if you'd like to export it to a CSV file
# synthetic_data.to_csv('10_rows.csv', index=False)
