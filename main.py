import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def load_data(filepath):
    """Load and return the Melbourne dataset."""
    return pd.read_csv(filepath)

def select_features(data, features):
    """Select and return the specified features from the dataset."""
    return data[features]

def train_model(features, target):
    """Train and return a Decision Tree model."""
    model = DecisionTreeRegressor(random_state=1)
    model.fit(features, target)
    return model

def make_prediction(model, input_data):
    """Make and return predictions using the trained model."""
    return model.predict(input_data)

def evaluate_model(model, features, actual_prices):
    """Evaluate the model and print the Mean Absolute Error."""
    predicted_prices = model.predict(features)
    print("Mean Absolute Error:", mean_absolute_error(actual_prices, predicted_prices))

if __name__ == "__main__":
    # Load and prepare data
    melb_file_path = './datasets/melb_data.csv'
    melb_data = load_data(melb_file_path)
    
    # Define features and target
    melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    melb_features_data = select_features(melb_data, melb_features)
    melb_data_prices = melb_data.Price

    # Train model
    melb_model = train_model(melb_features_data, melb_data_prices)

    # Make predictions on the first 5 houses
    print("Making predictions for the following 5 houses:")
    print(melb_features_data.head())
    predictions = make_prediction(melb_model, melb_features_data.head())
    print("The predictions are:", predictions)

    # Manual testing of the model
    example_input = {
        'Rooms': [2],
        'Bathroom': [1],
        'Landsize': [202],
        'Lattitude': [-37.7996],
        'Longtitude': [144.9984]
    }
    input_df = pd.DataFrame(example_input)
    predicted_price = make_prediction(melb_model, input_df)
    print(f"The predicted price of the house is: ${predicted_price[0]:,.2f}")

    # Evaluate model
    evaluate_model(melb_model, melb_features_data, melb_data_prices)

    print(melb_model)