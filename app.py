from flask import Flask, request, jsonify
from flask_cors import CORS
from model import get_stock_prediction_result  # Updated function from model.py

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from your React front end

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        stock_symbol = data.get('stock_symbol')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Basic validation
        if not stock_symbol or not start_date or not end_date:
            return jsonify({'error': 'Missing parameters'}), 400
        
        # Call the prediction function that returns the full data structure
        prediction_result = get_stock_prediction_result(stock_symbol, start_date, end_date)

        print("prediction_result",get_stock_prediction_result("AAPL","2023-01-01","2023-02-01"))
        print({'prediction': prediction_result})
        return jsonify(prediction_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)