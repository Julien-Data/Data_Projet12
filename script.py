# script_model_billets.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --------------------------------------
# 1. Chargement et nettoyage des données
# --------------------------------------

def charger_donnees(path):
    df = pd.read_csv(path, sep=';')  # Ajout du bon séparateur
    return df


def completer_margin_low_par_regression(df):
    df = df.copy()
    train_data = df.dropna(subset=['margin_low'])
    missing_data = df[df['margin_low'].isna()]
    features = ['length', 'margin_up', 'height_right']
    model = LinearRegression()
    X_train = train_data[features]
    y_train = train_data['margin_low']
    model.fit(X_train, y_train)
    X_missing = missing_data[features]
    predicted_values = model.predict(X_missing)
    df.loc[df['margin_low'].isna(), 'margin_low'] = predicted_values
    return df

# --------------------------------------
# 2. Préparation des données
# --------------------------------------

def preparer_donnees(df):
    X = df.drop(columns='is_genuine')
    y = df['is_genuine'].astype(int)  # bool → int (0/1)
    return X, y

def split_train_test(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --------------------------------------
# 3. Modèles
# --------------------------------------

def entrainer_modele(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluer_modele(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return accuracy_score(y_test, y_pred)

def evaluer_kmeans(df, X_scaled):
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    df['kmeans_pred'] = labels
    mapped = 1 - df['kmeans_pred'] if df['is_genuine'].corr(df['kmeans_pred']) < 0 else df['kmeans_pred']
    score = accuracy_score(df['is_genuine'], mapped)
    return score

def predire_billet(model, scaler, features_dict):
    """
    Prédit si un billet est vrai ou faux à partir de ses caractéristiques.
    """
    df = pd.DataFrame([features_dict])
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    return "VRAI billet" if prediction[0] == 1 else "FAUX billet"



def afficher_matrice_confusion(model, scaler, df):
    """
    Affiche la matrice de confusion du modèle donné.
    """
    df = df.copy()
    df = df.dropna(subset=['margin_low'])  # éviter les NaN
    X = df.drop(columns='is_genuine')
    y = df['is_genuine'].astype(int)
    
    X_scaled = scaler.transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title("Matrice de confusion")
    plt.show()

# --------------------------------------
# 4. Pipeline principal
# --------------------------------------

def pipeline_global(
    df=None,
    csv_path="billets.csv",
    test_size=0.2,
    random_state=42,
    use_scaler=True
):
    # 1. Chargement des données si non fourni
    if df is None:
        df = charger_donnees(csv_path)
    
    # 2. Préparation des données
    X, y = preparer_donnees(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    # 3. Standardisation
    if use_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = None
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # 4. Modèles supervisés
    logreg = entrainer_modele(LogisticRegression(), X_train_scaled, y_train)
    knn = entrainer_modele(KNeighborsClassifier(n_neighbors=5), X_train_scaled, y_train)
    rf = entrainer_modele(RandomForestClassifier(random_state=random_state), X_train_scaled, y_train)
    
    print("Régression Logistique:")
    acc_log = evaluer_modele(logreg, X_test_scaled, y_test)
    
    print("KNN:")
    acc_knn = evaluer_modele(knn, X_test_scaled, y_test)
    
    print("Random Forest:")
    acc_rf = evaluer_modele(rf, X_test_scaled, y_test)
    
    # 5. KMeans
    scaler_kmeans = StandardScaler()
    X_scaled = scaler_kmeans.fit_transform(X)
    acc_kmeans = evaluer_kmeans(df, X_scaled)
    print("K-Means accuracy (approx):", acc_kmeans)
    
    # 6. Résumé
    models = {
        "Régression Logistique": acc_log,
        "KNN": acc_knn,
        "Random Forest": acc_rf,
        "K-Means": acc_kmeans
    }
    
    print("\nRésumé des performances :")
    for name, score in models.items():
        print(f"{name}: {score:.4f}")
    
    # 7. Choix du modèle final
    best_model = rf
    
    return best_model, scaler





