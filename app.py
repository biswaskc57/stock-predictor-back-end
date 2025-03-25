from flask import Flask, request, jsonify
from flask_cors import CORS
from model import get_stock_prediction_result  # Your custom model function

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'Backend is running'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        stock_symbol = data.get('stock_symbol')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not stock_symbol or not start_date or not end_date:
            return jsonify({'error': 'Missing parameters'}), 400
        
        prediction_result = get_stock_prediction_result(stock_symbol, start_date, end_date)
        print("prediction_result", prediction_result)
        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
