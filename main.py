import pandas as pd
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
    melb_file_path = './datasets/melb_data.csv'
    melb_data = pd.read_csv(melb_file_path)
    
    # Selecting all of the column names
    melb_data_cols = melb_data.columns
    
    # Selecting the specific values in a column
    melb_data_prices = melb_data.Price

    # Choosing features
    melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    melb_features_data = melb_data[melb_features]

    melb_model = DecisionTreeRegressor(random_state=1)
    melb_model.fit(melb_features_data, melb_data_prices)
    print("Making predictions for the following 5 houses:")
    print(melb_features_data.head())
    print("The predictions are")
    print(melb_model.predict(melb_features_data.head()))