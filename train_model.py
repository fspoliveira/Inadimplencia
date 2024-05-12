import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
import joblib

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
    """Treina o modelo e salva o modelo."""
    # Balanceamento de classes usando oversampling
    oversampler = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    # Definindo o pipeline com imputação e modelo RandomForestClassifier
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])

    # Tuning de Hiperparâmetros
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [None, 10],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_model = grid_search.best_estimator_

    # Salvando o modelo
    joblib.dump(best_model, filepath)

    # Avaliando o modelo
    predictions = best_model.predict(X_train_resampled)
    print("Classification Report:\n", classification_report(y_train_resampled, predictions))
    print("ROC AUC score:", roc_auc_score(y_train_resampled, best_model.predict_proba(X_train_resampled)[:, 1]))

def main():
    """Função principal para executar as etapas do processo."""
    data = load_data('dataset/loan_default.csv')  # Certifique-se que o caminho está correto
    data_processed = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data_processed)
    train_and_save_model(X_train, y_train)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
