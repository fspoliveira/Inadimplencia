from flask import Flask, request
from flask_restx import Api, Resource, fields
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
api = Api(app, version='1.0', title='API de Predição',
          description='Uma API para fazer previsões usando um modelo de machine learning')

ns = api.namespace('Prediction', description='Operações de previsão')

# Modelo esperado para entrada de dados
model_input = api.model('ModelInput', {
    'loan_amount': fields.Integer(required=True, description='Quantia do empréstimo'),
    'Credit_Score': fields.Integer(required=True, description='Pontuação de crédito'),
    'loan_purpose': fields.String(required=True, description='Propósito do empréstimo'),
    'annual_income': fields.Float(required=True, description='Renda anual'),
    'term': fields.Integer(required=True, description='Duração do empréstimo em meses'),
    'rate_of_interest': fields.Float(required=True, description='Taxa de juros'),
    'age': fields.Integer(required=True, description='Idade do solicitante'),
    'employment_status': fields.String(required=True, description='Status de emprego'),
    'housing_status': fields.String(required=True, description='Status de moradia')
})

# Load model and scaler
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
model_columns = joblib.load('models/model_columns.pkl')

@ns.route('/predict')
class Predict(Resource):
    @api.expect(model_input)
    def post(self):
        """Recebe dados JSON e retorna a predição do modelo"""
        try:
            json_ = request.json
            query_df = pd.DataFrame([json_])
            query_df = query_df.reindex(columns=model_columns, fill_value=0)
            query_scaled = scaler.transform(query_df)
            prediction = model.predict(query_scaled)
            prediction_list = [int(i) for i in prediction.tolist()]
            return {'prediction': prediction_list}
        except Exception as e:
            api.abort(500, f'Erro ao fazer predição: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
