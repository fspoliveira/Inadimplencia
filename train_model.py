import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def load_data(filepath):
    """Carrega dados de um arquivo CSV."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """Pre-processa os dados substituindo valores ausentes e convertendo categorias em variáveis dummy."""
    data.ffill(inplace=True)
    data = pd.get_dummies(data, drop_first=True)
    return data

def split_data(data):
    """Divide os dados em conjuntos de treino e teste."""
    X = data.drop('Status', axis=1)
    y = data['Status'].astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_save_model(X_train, y_train, filepath='models/model.pkl'):
    """Treina o modelo e salva o modelo, o scaler e as colunas."""
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    # Usando balanceamento de classes
    class_weight = 'balanced'  # Isso ajusta os pesos inversamente proporcionais às frequências de classe
    model = LogisticRegression(class_weight=class_weight, max_iter=1000)

    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_train_scaled)

    print("Classification Report:\n", classification_report(y_train, predictions))
    print("ROC AUC score:", roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1]))

    # Validando o modelo com validação cruzada
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print("Cross-validated AUC scores:", scores)

    # Salvando o modelo, scaler e colunas
    joblib.dump(model, filepath)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')
    joblib.dump(X_train.columns, 'models/model_columns.pkl')

def main():
    """Função principal para executar as etapas do processo."""
    data = load_data('dataset/loan_default.csv')  # Certifique-se que o caminho está correto
    data_processed = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data_processed)
    train_and_save_model(X_train, y_train)
    print("Model, scaler, and columns saved successfully.")

if __name__ == "__main__":
    main()
