import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold

# Step 1: Combine Rare Categories
def group_events(event):
    holidays = [
        "New Year", "Women's Day", "Men's Day", "Independence Day of Ukraine",
        "Constitution Day of Ukraine", "Day of Defenders of Ukraine",
        "Valentine's Day", "Teacher's Day", "Day of Lviv city",
        "Day of Dignity and Freedom", "Day of Ukrainian Language",
        "The Nativity of Christ", "Saint Nicholas Day", "Easter"
    ]
    promotions = ["Special Promotion", "Seasonal Event"]
    occasions = ["Birthdays", "Corporate Event"]

    if event in holidays:
        return "Holiday"
    elif event in promotions:
        return "Promotion"
    elif event in occasions:
        return "Daily event"
    elif event is None:
        return "None"

def preprocess_data(file_path, is_event):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert 'Date' into separate Year, Month, Day columns
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Year', 'Month', 'Day']] = df['Date'].apply(lambda x: [x.year, x.month, x.day]).to_list()

    # Helper function for OneHot Encoding
    def onehot_encode(df, columns, prefix):
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded = encoder.fit_transform(df[columns])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(prefix), index=df.index)
        return encoded_df

    # OneHot encode 'Product'
    product_encoded_df = onehot_encode(df, ['Product'], ['Product'])

    # OneHot encode other categorical columns
    categorical_columns = ['Day_of_Week', 'Season', 'Weather', 'Category']
    categorical_encoded_df = onehot_encode(df, categorical_columns, categorical_columns)

    # Concatenate all encoded data
    df = pd.concat([df, product_encoded_df, categorical_encoded_df], axis=1)

    # Drop original categorical columns and 'Date'
    df = df.drop(['Day_of_Week', 'Season', 'Weather', 'Product', 'Date', 'Category'], axis=1)

    if is_event:
        # Group and encode 'Event'
        df['Event_Grouped'] = df['Event'].apply(group_events)
        event_encoded_df = onehot_encode(df, ['Event_Grouped'], ['Event_Grouped'])

        # Concatenate the encoded Event Data
        df = pd.concat([df, event_encoded_df], axis=1)

        # Drop the original 'Event' and grouped column
        df = df.drop(['Event', 'Event_Grouped'], axis=1)
    else:
        df = df.drop(columns='Event')

    # Split features and target
    X = df.drop(['Purchase_Quantity'], axis=1)  # Features
    y = df['Purchase_Quantity']  # Target

    # Standardize the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

    return X_scaled, y, kf


def preprocess_test_data(file_path, is_event):
    # Step 1: Load the dataset
    df = pd.read_csv(file_path)

    # Step 2: Convert 'Date' into separate Year, Month, Day columns
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Year', 'Month', 'Day']] = df['Date'].apply(lambda x: [x.year, x.month, x.day]).to_list()

    def onehot_encode(df, columns, prefix):
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded = encoder.fit_transform(df[columns])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(prefix), index=df.index)
        return encoded_df

    # Step 3: Label encode 'Product'
    product_encoded_df = onehot_encode(df, ['Product'], ['Product'])

    # Step 4: OneHot encode other categorical columns
    train_categories = {
        'Day_of_Week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'Season': ['Winter', 'Spring', 'Summer', 'Autumn'],
        'Weather': ['Sunny', 'Rainy', 'Snowy', 'Cloudy', 'Stormy', 'Hot', 'Cold'],
        'Category': ['Dairy', 'Meat', 'Vegetables', 'Fruits', 'Seafood'],
    }
    train_onehot_encoder = OneHotEncoder(categories=list(train_categories.values()), drop='first', sparse_output=False)
    encoded_columns = train_onehot_encoder.fit_transform(df[['Day_of_Week', 'Season', 'Weather', 'Category']])
    encoded_column_names = train_onehot_encoder.get_feature_names_out(['Day_of_Week', 'Season', 'Weather', 'Category'])
    encoded_df = pd.DataFrame(encoded_columns, columns=encoded_column_names, index=df.index)

    # Step 5: Concatenate the original DataFrame with the encoded DataFrame
    df = pd.concat([df, product_encoded_df, encoded_df], axis=1)

    # Step 6: Drop the original categorical columns and 'Date'
    df = df.drop(['Day_of_Week', 'Season', 'Weather', 'Product', 'Date', 'Category'], axis=1)
    if is_event != 0:
        event_category = {
            'Event_Grouped': ["Holiday", "Daily event", "Promotion", "None"]
        }
        train_onehot_encoder = OneHotEncoder(categories=list(event_category.values()), drop='first', sparse_output=False)
        encoded_columns = train_onehot_encoder.fit_transform(df[['Event']])
        encoded_column_names = train_onehot_encoder.get_feature_names_out(['Event'])
        encoded_event = pd.DataFrame(encoded_columns, columns=encoded_column_names, index=df.index)

        df = pd.concat([df, encoded_event], axis=1)
        df = df.drop(columns='Event')
    else:
        df = df.drop(columns='Event')

    # Step 10: Extract target variable
    y = df['Purchase_Quantity'].copy()
    df = df.drop(['Purchase_Quantity'], axis=1).copy()

    # Step 11: Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    return X_scaled, y
