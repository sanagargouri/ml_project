import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(file_path):
    """
    Charge le jeu de données à partir d'un fichier CSV.

    Parameters
    ----------
    file_path : str
        Chemin vers le fichier CSV.

    Returns
    -------
    pandas.DataFrame
        Jeu de données chargé.
    """
    data = pd.read_csv(file_path)
    return data

def prepare_data(data):
    """
    Prépare les données pour l'entraînement du modèle.

    Parameters
    ----------
    data : pandas.DataFrame
        Jeu de données brut.

    Returns
    -------
    X : pandas.DataFrame
        Variables explicatives.
    y : pandas.Series
        Variable cible.
    """
    data = data.copy()

    # Encoder la variable Gender
    label_encoder = LabelEncoder()
    data["Gender"] = label_encoder.fit_transform(data["Gender"])

    # Supprimer les colonnes inutiles
    data = data.drop(["Surname", "Geography"], axis=1)

    # Séparer les variables explicatives et la cible
    X = data.drop(["Exited"], axis=1)
    y = data["Exited"]

    # Supprimer les identifiants
    X = X.drop(["RowNumber", "CustomerId"], axis=1)

    return X, y

def split_data(X, y, test_size=0.2, random_state=1):
    """
    Sépare les données en ensembles d'entraînement et de test.

    Parameters
    ----------
    X : pandas.DataFrame
        Variables explicatives.
    y : pandas.Series
        Variable cible.
    test_size : float, optional
        Proportion des données réservées au test.
    random_state : int, optional
        Graine utilisée pour rendre la séparation reproductible.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Ensembles d'entraînement et de test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Crée et entraîne un modèle Random Forest.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Variables explicatives d'entraînement.
    y_train : pandas.Series
        Variable cible d'entraînement.

    Returns
    -------
    RandomForestClassifier
        Modèle Random Forest entraîné.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

def predict(model, X_test):
    """
    Effectue des prédictions avec le modèle entraîné.

    Parameters
    ----------
    model : RandomForestClassifier
        Modèle entraîné.
    X_test : pandas.DataFrame
        Données de test.

    Returns
    -------
    numpy.ndarray
        Prédictions du modèle.
    """
    y_pred = model.predict(X_test)

    return y_pred


def evaluate_model(y_test, y_pred):
    """
    Évalue les performances du modèle.

    Parameters
    ----------
    y_test : pandas.Series
        Valeurs réelles.
    y_pred : numpy.ndarray
        Prédictions du modèle.

    Returns
    -------
    float
        Accuracy du modèle.

    numpy.ndarray
        Matrice de confusion.
    """
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm

def save_model(model, file_path):
    """
    Sauvegarde un modèle entraîné dans un fichier.

    Parameters
    ----------
    model : RandomForestClassifier
        Modèle entraîné à sauvegarder.
    file_path : str
        Chemin du fichier de sauvegarde.
    """
    joblib.dump(model, file_path)

def load_model(file_path):
    """
    Charge un modèle sauvegardé.

    Parameters
    ----------
    file_path : str
        Chemin du fichier contenant le modèle sauvegardé.

    Returns
    -------
    RandomForestClassifier
        Modèle chargé.
    """
    model = joblib.load(file_path)

    return model

