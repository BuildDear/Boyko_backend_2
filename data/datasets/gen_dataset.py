import pandas as pd
from additional_functions import *
from datetime import datetime


# Parameters for data generation
products = ['Milk', 'Eggs', 'Chicken', 'Tomatoes', 'Apples', 'Salmon', 'Cheese', 'Lettuce', 'Pork', 'Potatoes']
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weather_conditions = ['Sunny', 'Rainy', 'Snowy', 'Cloudy', 'Stormy', 'Hot', 'Cold']


# Generate synthetic data
def generate_synthetic_data(is_event, n_rows):
    data = []

    # Date range for purchases
    current_date = datetime(2004, 1, 1)
    end_date = datetime(2023, 12, 31)
    # Track the last purchase date for each product to ensure weekly purchases
    product_purchase_tracker = {product: current_date for product in products}

    for _ in range(n_rows):

        holiday_name = None
        product = random.choice(products)

        if is_event == 0:
            current_date = generate_sequential_date(current_date, product_purchase_tracker, product)
        else:
            current_date, holiday_name = generate_date_with_event(current_date, product_purchase_tracker, product)
        year = current_date.year

        # Ensure the date doesn't exceed the end_date
        if current_date > end_date:
            break  # Stop generation if we've reached the end date

        season = determine_season(current_date)
        weather = random.choice(seasonal_weather[season])
        day_of_week = days_of_week[current_date.weekday()]

        if is_event == 0:
            event = None
        else:
            if holiday_name is None:
                event = get_event()
            else:
                event = holiday_name

        num_customers = generate_num_customers(current_date, season, weather)
        stocks = get_stock(num_customers, event)
        days_until_next_purchase = next_purchase(stocks, product)
        category = get_category(product)
        quantity = determine_quantity(num_customers, stocks, season, product, event)
        unit_price = get_price(season, product, year)
        sales = get_average_check(num_customers)
        shelf_life = get_shelf_life(product)

        # Append row data
        data.append([current_date, day_of_week, season, weather, product, category, unit_price,
                     num_customers, sales, stocks, shelf_life, days_until_next_purchase, event, quantity])

    # Create DataFrame
    columns = ['Date', 'Day_of_Week', 'Season', 'Weather', 'Product', 'Category', 'Unit_Price', 'Num_Customers',
               'Sales', 'Stocks', 'Shelf_Life', 'Days_Until_Next_Purchase', 'Event', 'Purchase_Quantity']

    df = pd.DataFrame(data, columns=columns)
    return df

is_event = 1

if is_event == 1:
    synthetic_data = generate_synthetic_data(is_event, 100000)
    synthetic_data.to_csv('data/datasets/dataset_event.csv', index=False)
else:
    synthetic_data = generate_synthetic_data(is_event, 100000)
    synthetic_data.to_csv('data/datasets/dataset.csv', index=False)

print(synthetic_data.head())

print("\nТипи даних кожного атрибуту:")
print(synthetic_data.dtypes)