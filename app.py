from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
model_columns = joblib.load('models/model_columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        query_df = pd.DataFrame([json_])
        query_df = query_df.reindex(columns=model_columns, fill_value=0)
        query_scaled = scaler.transform(query_df)
        prediction = model.predict(query_scaled)
        # Converta numpy array para lista de inteiros nativos do Python
        prediction_list = prediction.tolist()
        # Assegura que todos os elementos são inteiros nativos do Python
        prediction_list = [int(i) for i in prediction_list]
        return jsonify({'prediction': prediction_list})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':   
    app.run(debug=True, host='localhost', port=5001)
