import pandas as pd
from additional_functions import *
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Parameters for data generation
products = ['Milk', 'Eggs', 'Chicken', 'Tomatoes', 'Apples', 'Salmon', 'Cheese', 'Lettuce', 'Pork', 'Potatoes']
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weather_conditions = ['Sunny', 'Rainy', 'Snowy', 'Cloudy', 'Stormy', 'Hot', 'Cold']

def corr_matrix(file_path, is_event):
   # Load the dataset
   df = pd.read_csv(file_path)

   # Convert 'Date' into separate Year, Month, Day columns
   df['Date'] = pd.to_datetime(df['Date'])
   df[['Year', 'Month', 'Day']] = df['Date'].apply(lambda x: [x.year, x.month, x.day]).to_list()

   np.random.seed(42)  # For reproducibility
   noise = df['Sales'] * 0.14 * np.random.normal(0, 1, size=len(df))
   df['Sales'] = df['Sales'] + noise

   # Drop original categorical columns and 'Date'
   df = df.drop(['Day_of_Week', 'Weather', 'Product', 'Date', 'Category', 'Event', 'Season'], axis=1)

   # Compute and plot the correlation matrix
   correlation_matrix = df.corr()
   plt.figure(figsize=(45, 40))
   sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=False, linewidths=0.5, annot_kws={"size": 8}, yticklabels=1)
   plt.title('Correlation Matrix After Preprocessing')
   plt.show()

   plt.yticks(rotation=0)
   plt.xticks(rotation=45)
   plt.tight_layout()

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

#
# is_event = 1
#
# if is_event == 1:
#     synthetic_data = generate_synthetic_data(is_event, 100000)
#     synthetic_data.to_csv('dataset_event.csv', index=False)
# else:
#     synthetic_data = generate_synthetic_data(is_event, 100000)
#     synthetic_data.to_csv('dataset.csv', index=False)


# print(synthetic_data.head())

############ corr matrix #######
# file_path = 'data/datasets/dataset.csv'
# corr_matrix(file_path, 0)




#def corr_matrix(file_path, is_event):
#    # Load the dataset
#    df = pd.read_csv(file_path)
#
#    # Convert 'Date' into separate Year, Month, Day columns
#    df['Date'] = pd.to_datetime(df['Date'])
#    df[['Year', 'Month', 'Day']] = df['Date'].apply(lambda x: [x.year, x.month, x.day]).to_list()
#
#    # Helper function for OneHot Encoding
#    def onehot_encode(df, columns, prefix):
#        encoder = OneHotEncoder(drop='first', sparse_output=False)
#        encoded = encoder.fit_transform(df[columns])
#        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(prefix), index=df.index)
#        return encoded_df
#
#    # OneHot encode 'Product'
#    product_encoded_df = onehot_encode(df, ['Product'], ['Product'])
#
#    label_encoder = LabelEncoder()
#    df['Season'] = label_encoder.fit_transform(df['Season'])
#
#    # OneHot encode other categorical columns
#    categorical_columns = ['Day_of_Week', 'Weather', 'Category']
#    categorical_encoded_df = onehot_encode(df, categorical_columns, categorical_columns)
#
#    # Concatenate all encoded data
#    df = pd.concat([df, product_encoded_df, categorical_encoded_df], axis=1)
#
#    # Drop original categorical columns and 'Date'
#    df = df.drop(['Day_of_Week', 'Weather', 'Product', 'Date', 'Category'], axis=1)
#
#    if is_event:
#        # Group and encode 'Event'
#        df['Event_Grouped'] = df['Event'].apply(group_events)
#        event_encoded_df = onehot_encode(df, ['Event_Grouped'], ['Event_Grouped'])
#
#        # Concatenate the encoded Event Data
#        df = pd.concat([df, event_encoded_df], axis=1)
#
#        # Drop the original 'Event' and grouped column
#        df = df.drop(['Event', 'Event_Grouped'], axis=1)
#    else:
#        df = df.drop(columns='Event')
#
#    # Split features and target
#    X = df.drop(['Purchase_Quantity'], axis=1)  # Features
#    y = df['Purchase_Quantity']  # Target
#
#    # Compute and plot the correlation matrix
#    correlation_matrix = X.corr()
#    plt.figure(figsize=(45, 40))
#    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=False, linewidths=0.5, annot_kws={"size": 8})
#    plt.title('Correlation Matrix After Preprocessing')
#    plt.show()
#
#    return X, y