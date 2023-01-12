from flask import Flask, jsonify, request
import yfinance as yf
from fbprophet import Prophet

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    stock = request.args.get('stock')
    if stock is None:
        return jsonify({"error": "Missing 'stock' parameter in the request"})

    # fetch historical data for the stock from yfinance
    data = yf.download(stock, start='2010-01-01', end='2023-12-31')


    # Formatting the dataframe to match the required format by Prophet Model
    data = data[['Close']]
    data.columns = ['y']
    data['ds'] = data.index
    
    # create a Prophet model
    m = Prophet()
    m.fit(data)

    # create a dataframe to hold the predictions
    future = m.make_future_dataframe(periods=365)


    forecast = m.predict(future)

    # extract the relevant prediction columns from the forecast dataframe
    prediction = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(365)

    # return the prediction data as a JSON object
    return jsonify(prediction.to_dict())

if __name__ == '__main__':
    app.run(debug=True)
