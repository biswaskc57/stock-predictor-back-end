import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def fetch_data(stock_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock data for the given symbol and date range using yfinance.
    """
    ticker = yf.Ticker(stock_symbol)
    data = ticker.history(start=start_date, end=end_date)
    return data

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by selecting the 'Close' price and converting dates to ordinal numbers.
    """
    if data.empty:
        raise ValueError("No data available for the given parameters.")
    
    # Select only the 'Close' column
    data = data[['Close']]
    data = data.reset_index()  # Ensure 'Date' is a column
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Convert the date to a numeric format for regression (ordinal)
    data['Date_ordinal'] = data['Date'].apply(lambda date: date.toordinal())
    return data

def train_model(data: pd.DataFrame) -> LinearRegression:
    """
    Train a linear regression model using the ordinal date as the predictor and 'Close' as the target.
    """
    X = np.array(data['Date_ordinal']).reshape(-1, 1)
    y = data['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_value(model: LinearRegression, date_ordinal: int) -> float:
    """
    Predict the stock price for a given date (in ordinal format) using the trained model.
    """
    prediction = model.predict(np.array([[date_ordinal]]))
    return float(prediction[0])

def get_stock_prediction_result(stock_symbol: str, start_date: str, end_date: str):
    """
    Fetch data, preprocess it, train the model, and return the final prediction data in the specified format.
    """
    # Fetch and preprocess historical data
    data = fetch_data(stock_symbol, start_date, end_date)
    data = preprocess_data(data)
    model = train_model(data)
    
    # Get the last historical date
    last_date = data['Date'].max()
    last_date_ordinal = last_date.toordinal()
    
    # Build historical data records using relative labels (for the last 8 days)
    historical_data = data.tail(8)
    historical_records = []
    relative_labels = ["1 Week Ago", "6 Days Ago", "5 Days Ago", "4 Days Ago", "3 Days Ago", "2 Days Ago", "Yesterday", "Today"]
    for i, (_, row) in enumerate(historical_data.iterrows()):
        date_label = relative_labels[i] if i < len(relative_labels) else row['Date'].strftime('%Y-%m-%d')
        actual_price = row['Close']
        predicted_price = predict_value(model, row['Date_ordinal'])
        historical_records.append({
            "date": date_label,
            "price": round(actual_price, 2),
            "prediction": round(predicted_price, 2)
        })
    
    # Build future prediction records for the next 3 days
    future_records = []
    future_labels = ["Tomorrow", "2 Days Later", "3 Days Later"]
    for i, label in enumerate(future_labels, start=1):
        future_date_ordinal = last_date_ordinal + i
        predicted_price = predict_value(model, future_date_ordinal)
        future_records.append({
            "date": label,
            "price": None,
            "prediction": round(predicted_price, 2)
        })
    
    # Combine historical and future records
    combined_data = historical_records + future_records
    
    # Additional aggregated predictions
    next_day = predict_value(model, last_date_ordinal + 1)
    next_week = predict_value(model, last_date_ordinal + 7)
    next_month = predict_value(model, last_date_ordinal + 30)
    
    result = {
        "symbol": stock_symbol,
        "data": combined_data,
        "prediction": {
            "nextDay": round(next_day, 2),
            "nextWeek": round(next_week, 2),
            "nextMonth": round(next_month, 2)
        }
    }
    return result
print("prediction_result",get_stock_prediction_result("AAPL","2023-01-01","2023-02-01"))