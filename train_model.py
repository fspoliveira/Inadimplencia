import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
import joblib


def load_data(filepath):
    """Carrega dados de um arquivo CSV."""
    data = pd.read_csv(filepath)
    return data


def preprocess_data(data):
    """Pre-processa os dados substituindo valores ausentes e convertendo categorias em variáveis dummy."""
    # Separa os recursos numéricos e categóricos
    numerical_features = data.select_dtypes(include=['float64', 'int64'])
    categorical_features = data.select_dtypes(include=['object'])

    # Imputa valores ausentes nos recursos numéricos com a média
    imputer_numerical = SimpleImputer(strategy='mean')
    numerical_imputed = imputer_numerical.fit_transform(numerical_features)

    # Converte o array resultante de volta em um DataFrame
    numerical_processed = pd.DataFrame(numerical_imputed, columns=numerical_features.columns)

    # Imputa valores ausentes nos recursos categóricos com a moda (valor mais comum)
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    categorical_imputed = imputer_categorical.fit_transform(categorical_features)

    # Converte o array resultante de volta em um DataFrame
    categorical_processed = pd.DataFrame(categorical_imputed, columns=categorical_features.columns)

    # Convertendo categorias em variáveis dummy para recursos categóricos
    categorical_processed = pd.get_dummies(categorical_processed, drop_first=True)

    # Combina os DataFrames processados
    data_processed = pd.concat([numerical_processed, categorical_processed], axis=1)

    return data_processed


def split_data(data):
    """Divide os dados em conjuntos de treino e teste."""
    X = data.drop('Status', axis=1)
    y = data['Status'].astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_save_model(X_train, y_train, filepath='models/model.pkl'):
    """Treina o modelo e salva o modelo."""
    model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Salvando o modelo
    joblib.dump(model, filepath)

    # Avaliando o modelo
    predictions = model.predict(X_train)
    print("Classification Report:\n", classification_report(y_train, predictions))
    print("ROC AUC score:", roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))


def main():
    """Função principal para executar as etapas do processo."""
    data = load_data('dataset/loan_default.csv')  # Certifique-se que o caminho está correto
    data_processed = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data_processed)
    train_and_save_model(X_train, y_train)
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
