import random
from datetime import datetime, timedelta

events = ['None', 'Birthdays', 'Corporate Event', 'Special Promotion', 'Seasonal Event']

# Список свят (без року)
holidays = [
    ("New Year", 1, 1),
    ("Women's Day", 3, 8),
    ("Men's Day", 11, 14),
    ("Independence Day of Ukraine", 8, 24),
    ("Constitution Day of Ukraine", 6, 28),
    ("Day of Defenders of Ukraine", 10, 14),
    ("Valentine's Day", 2, 14),
    ("Teacher's Day", 10, 5),
    ("Day of Lviv city", 4, 27),
    ("Day of Dignity and Freedom", 11, 21),
    ("Day of Ukrainian Language", 11, 9),
    ("The Nativity of Christ", 12, 25),
    ("Saint Nicholas Day", 12, 6),
    ("Easter", 4, 16),  # Дату Великодня потрібно розраховувати кожного року
]


def generate_sequential_date(previous_date, product_purchase_tracker, product, min_days=0, max_days=3):
    """
    Generate a sequential date for a product, ensuring that it is purchased at least once a week.
    The function tracks the last purchase date for each product and ensures a purchase every 7 days.
    """
    last_purchase_date = product_purchase_tracker.get(product, previous_date)

    # Add a random number of days between min_days and max_days to ensure purchases are spaced out
    delta_days = random.randint(min_days, max_days)

    # Calculate new purchase date
    new_date = last_purchase_date + timedelta(days=delta_days)

    # Ensure the new date is not more than 4 days from the last purchase
    if (new_date - last_purchase_date).days > 4:
        new_date = last_purchase_date + timedelta(days=4)

    # Update the tracker with the new purchase date for the product
    product_purchase_tracker[product] = new_date

    return new_date


def generate_date_with_event(previous_date, product_purchase_tracker, product, min_days=0, max_days=3):
    """
    Генерує послідовну дату для продукту, враховуючи святкові дати.
    Закупівля відбувається за день до свята.
    """
    last_purchase_date = product_purchase_tracker.get(product, previous_date)

    # Генерація випадкової кількості днів між мінімумом та максимумом
    delta_days = random.randint(min_days, max_days)

    # Обчислення нової дати покупки
    new_date = last_purchase_date + timedelta(days=delta_days)

    # Отримання святкових дат для поточного року
    holiday_dates = [
        (datetime(new_date.year, month, day), name) for name, month, day in holidays
    ]

    holiday_name = None
    for holiday_date, name in holiday_dates:
        if new_date <= holiday_date <= new_date + timedelta(days=max_days):
            if holiday_date.month == 1 and holiday_date.day == 1:
                new_date = datetime(holiday_date.year - 1, 12, 31)
            else:
                new_date = holiday_date - timedelta(days=1)
            holiday_name = name
            break

    product_purchase_tracker[product] = new_date

    if holiday_name:
        return new_date, holiday_name
    else:
        return new_date, None


def get_price(season, product, year):
    # Base prices in 2004
    base_prices = {
        'Milk': 4,
        'Lettuce': 2,
        'Eggs': 4,
        'Cheese': 20,
        'Chicken': 15,
        'Tomatoes': 5,
        'Apples': 2,
        'Salmon': 16,
        'Pork': 16,
        'Potatoes': 1,
    }

    # Annual inflation rates
    annual_inflation = [1.123, 1.103, 1.116, 1.166, 1.223, 1.123, 1.091, 1.046, 0.998, 1.005, 1.249,
                        1.433, 1.124, 1.137, 1.098, 1.041, 1.05, 1.1, 1.266, 1.051, 1.065]  # Last element adjusted

    # Calculate the cumulative inflation multiplier from 2004 to the specified year
    inflation_multiplier = 1.0
    for i in range(year - 2004):  # Limit to the number of years in annual_inflation
        inflation_multiplier *= annual_inflation[i]

    # Adjust base prices for the current year
    adjusted_price = base_prices.get(product, 0) * inflation_multiplier

    # Seasonal adjustments
    if season == 'Summer':
        seasonal_adjustment = {
            'Milk': 1.2,
            'Lettuce': 2.2,
            'Eggs': 1.5,
            'Cheese': 1.1,
            'Chicken': 1.2,
            'Tomatoes': 2.0,
            'Apples': 1.5,
            'Salmon': 2.0,
            'Pork': 1.3,
            'Potatoes': 1.2,
        }
    elif season == 'Winter':
        seasonal_adjustment = {
            'Milk': 1.3,
            'Lettuce': 3.0,
            'Eggs': 1.8,
            'Cheese': 1.4,
            'Chicken': 1.3,
            'Tomatoes': 2.5,
            'Apples': 2.2,
            'Salmon': 2.3,
            'Pork': 1.6,
            'Potatoes': 1.4,
        }
    else:  # For Spring and Autumn
        seasonal_adjustment = {
            'Milk': 1.1,
            'Lettuce': 1.6,
            'Eggs': 1.3,
            'Cheese': 1.2,
            'Chicken': 1.1,
            'Tomatoes': 1.6,
            'Apples': 1.2,
            'Salmon': 1.7,
            'Pork': 1.4,
            'Potatoes': 1.1,
        }

    # Apply seasonal adjustment
    final_price = adjusted_price * seasonal_adjustment.get(product, 1.0)
    return round(final_price, 2)


# Dictionary to map products to their categories
product_category = {
    'Milk': 'Dairy',
    'Eggs': 'Dairy',
    'Chicken': 'Meat',
    'Pork': 'Meat',
    'Tomatoes': 'Vegetables',
    'Apples': 'Fruits',
    'Salmon': 'Seafood',
    'Cheese': 'Dairy',
    'Lettuce': 'Vegetables',
    'Potatoes': 'Vegetables'
}

# Shelf life dictionary (in days)
shelf_life_dict = {
    'Milk': 7,       # milk can last around 7 days
    'Eggs': 21,      # eggs can last about 3 weeks
    'Chicken': 5,    # chicken can be stored for up to 5 days
    'Tomatoes': 10,  # tomatoes last about 10 days
    'Apples': 30,    # apples can last for a month
    'Salmon': 3,       # fresh fish lasts about 3 days
    'Cheese': 9,    # hard cheese can last about 90 days
    'Lettuce': 5,    # lettuce lasts around 5 days
    'Pork': 7,       # beef can last up to a week
    'Potatoes': 9   # potatoes can be stored for several months
}


# Function to get the correct category for each product
def get_category(product):
    return product_category.get(product, 'Other')  # Returns 'Other' if product not found


def get_shelf_life(product):
    return shelf_life_dict[product]


# Weights: 40% for 'None', 60% distributed across other events
event_weights = [0.4, 0.15, 0.15, 0.15, 0.15]


# Choose an event based on the specified probabilities
def get_event():
    return random.choices(events, weights=event_weights, k=1)[0]


def get_stock(num_customers, event):
    if num_customers > 600 or event is not None:
        stock_left = random.randint(1, 30)  # Large stock if few visitors
        return stock_left
    else:
        stock_left = random.randint(20, 70)  # Small stock if more visitors
        return stock_left


def get_average_check(num_customers):
    checks = []
    for _ in range(num_customers):
        checks.append(random.uniform(200, 1000))
    return sum(checks)


def determine_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'


# Weather conditions based on the season
seasonal_weather = {
    'Winter': ['Snowy', 'Cold', 'Cloudy'],
    'Spring': ['Sunny', 'Rainy', 'Cloudy'],
    'Summer': ['Sunny', 'Hot', 'Cloudy'],
    'Autumn': ['Rainy', 'Cloudy', 'Windy']
}


def next_purchase(stock_left, product):
    if stock_left < 10:
        days_until_next_purchase = 1  # Restock immediately if stock is low
        return days_until_next_purchase
    else:
        days_until_next_purchase = get_shelf_life(product)  # Otherwise, use shelf life
        return days_until_next_purchase


# Function to generate number of customers based on various factors
def generate_num_customers(current_date, season, weather):
    """
    Calculate the number of customers from start_date to end_date, considering
    seasonal and weather conditions for each day.
    """
    total_customers = 0

    for _ in range(30):
        base_customers = 250  # Base number of visitors in average conditions

        # Adjust for season
        if season == 'Summer':
            base_customers += random.randint(10, 100)  # More visitors in summer
        elif season == 'Winter':
            base_customers -= random.randint(10, 50)  # Fewer visitors in winter

        # Adjust for weather
        if weather == 'Sunny':
            base_customers += random.randint(10, 50)  # More visitors in sunny weather
        elif weather in ['Rainy', 'Snowy', 'Stormy']:
            base_customers -= random.randint(10, 100)  # Fewer visitors in bad weather

        # Adjust for weekends
        day_of_week = current_date.weekday()
        if day_of_week in [4, 5, 6]:  # Friday, Saturday, Sunday
            base_customers += random.randint(20, 100)

        # Generate final number of customers for the day with some randomness
        num_customers = random.randint(base_customers - 50, base_customers + 50)

        # Ensure number of customers stays within realistic bounds
        num_customers = max(50, min(800, num_customers))
        total_customers += num_customers

        # Move to the next day
        current_date += timedelta(days=1)

    return total_customers


def num_customers_events(current_date, season, weather):
    """
    Calculate the number of customers from start_date to end_date, considering
    seasonal and weather conditions for each day.
    """
    total_customers = 0

    for i in range(30):
        base_customers = 250  # Base number of visitors in average conditions

        # Adjust for season
        if season == 'Summer':
            base_customers += random.randint(10, 100)  # More visitors in summer
        elif season == 'Winter':
            base_customers -= random.randint(10, 50)  # Fewer visitors in winter

        # Adjust for weather
        if weather == 'Sunny':
            base_customers += random.randint(10, 50)  # More visitors in sunny weather
        elif weather in ['Rainy', 'Snowy', 'Stormy']:
            base_customers -= random.randint(10, 100)  # Fewer visitors in bad weather

        # Adjust for weekends
        day_of_week = current_date.weekday()
        if day_of_week in [4, 5, 6]:  # Friday, Saturday, Sunday
            base_customers += random.randint(20, 100)

        event = i % 2

        if event == 1:
            base_customers += random.randint(5, 30)

        # Generate final number of customers for the day with some randomness
        num_customers = random.randint(base_customers - 50, base_customers + 50)

        # Ensure number of customers stays within realistic bounds
        num_customers = max(50, min(800, num_customers))
        total_customers += num_customers

        # Move to the next day
        current_date += timedelta(days=1)

    return total_customers


# New logic to determine purchase quantity
def determine_quantity(num_customers, stocks, season, product, event):
    base_quantity = 30  # Base quantity for all products

    # Increase quantity if many customers
    if num_customers > 6500:
        base_quantity += random.randint(10, 15)  # More customers, larger purchase

    # Increase quantity if low stock
    if stocks < 10:
        base_quantity += 10  # Urgent need to restock

    # Seasonal adjustments
    if season == 'Summer':
        if product in ['Tomatoes', 'Lettuce', 'Apples', 'Chicken']:
            base_quantity += 5  # High demand for fresh produce
        elif product == 'Chicken':
            base_quantity += 10

    elif season == 'Winter':
        if product in ['Milk', 'Cheese', 'Pork', 'Potatoes']:
            base_quantity += 5  # Comfort foods and warming dishes
        elif product == 'Potatoes':
            base_quantity += 10

    elif season == 'Spring':
        if product in ['Lettuce', 'Tomatoes', 'Eggs', 'Salmon']:
            base_quantity += 5  # Demand for light, fresh foods
        elif product == 'Salmon':
            base_quantity += 10

    elif season == 'Autumn':
        if product in ['Apples', 'Pork', 'Potatoes', 'Chicken']:
            base_quantity += 5  # Seasonal fruits and hearty foods
        elif product == 'Chicken':
            base_quantity += 10

    if event is not None:
        base_quantity += random.randint(9, 13)

    return base_quantity
