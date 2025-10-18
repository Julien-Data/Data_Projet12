# script_model_billets.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from statsmodels.stats.outliers_influence import variance_inflation_factor
# --------------------------------------
# 1. Chargement et nettoyage des données
# --------------------------------------

def charger_donnees(path):
    df = pd.read_csv(path, sep=';')  # Ajout du bon séparateur
    return df



def verifier_hypotheses_regression(df):

    df = df.dropna(subset=['margin_low'])
    X = df[['length', 'margin_up', 'height_right']]
    y = df['margin_low']
    
    X = sm.add_constant(X)  # Ajoute l'intercept
    model = sm.OLS(y, X).fit()
    
    print(model.summary())

    # Résidus
    residus = model.resid
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Normalité des résidus
    sns.histplot(residus, kde=True, ax=axes[0])
    axes[0].set_title("Histogramme des résidus")

    sm.qqplot(residus, line='45', fit=True, ax=axes[1])
    axes[1].set_title("Q-Q plot (normalité des résidus)")

    # Homoscédasticité
    axes[2].scatter(model.fittedvalues, residus)
    axes[2].axhline(0, color='red', linestyle='--')
    axes[2].set_title("Résidus vs valeurs ajustées")
    axes[2].set_xlabel("Valeurs ajustées")
    axes[2].set_ylabel("Résidus")

    plt.tight_layout()
    plt.show()


def calculer_vif(df, features):
    X = df[features].dropna()
    vif_data = pd.DataFrame()
    vif_data["variable"] = features
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(features))]
    print("Variance Inflation Factor (VIF) :")
    print(vif_data)


def completer_margin_low_par_regression(df):
    df = df.copy()
    train_data = df.dropna(subset=['margin_low'])
    missing_data = df[df['margin_low'].isna()]
    features = ['length', 'margin_up', 'height_right']
    
    model = LinearRegression()
    X_train = train_data[features]
    y_train = train_data['margin_low']
    model.fit(X_train, y_train)

    # 💡 Ajout pour affichage interprétatif
    print(" Régression linéaire pour imputation de margin_low")
    print(f"R² : {model.score(X_train, y_train):.4f}")
    for f, c in zip(features, model.coef_):
        print(f"  Coefficient pour {f}: {c:.4f}")
    print(f"  Intercept : {model.intercept_:.4f}")
    print("-" * 40)

    # Imputation
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
    Garantit que l'ordre et les colonnes correspondent à ceux du scaler.
    """
    df = pd.DataFrame([features_dict])

    # S'assurer que les colonnes sont dans le bon ordre
    if hasattr(scaler, 'feature_names_in_'):
        colonnes_attendues = scaler.feature_names_in_
        df = df[colonnes_attendues]
    else:
        # Cas où feature_names_in_ n’existe pas (selon version sklearn)
        raise ValueError("Impossible de retrouver les colonnes utilisées pour le fit du scaler.")

    # Transformation et prédiction
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)

    return "VRAI billet" if prediction[0] == 1 else "FAUX billet"




def afficher_matrice_confusion(model, scaler, df):
    """
    Affiche la matrice de confusion du modèle donné.
    """
    df = df.copy()
    df = df.dropna(subset=['margin_low'])  # éviter les NaN
    
    # Récupérer les colonnes utilisées lors du fit du scaler
    colonnes_modele = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else df.drop(columns='is_genuine').columns
    
    # Enlever les colonnes non présentes dans le scaler s'il y en a (ex: kmeans_pred)
    colonnes_a_utiliser = [col for col in colonnes_modele if col in df.columns]
    
    X = df[colonnes_a_utiliser]
    y = df['is_genuine'].astype(int)
    
    X_scaled = scaler.transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title("Matrice de confusion")
    plt.show()




def validation_croisee_modele(model, X, y, cv=5):
    """
    Effectue une validation croisée avec plusieurs métriques (accuracy, precision, recall, f1).
    Affiche la moyenne et l'écart-type pour chaque métrique.
    """
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    results = cross_validate(model, X, y, cv=cv, scoring=scoring) # partie avec stratification

    print(f"Validation croisée ({cv} folds) - Moyennes des scores :\n")
    for metric in scoring:
        scores = results[f'test_{metric}']
        print(f"{metric:<15}: {scores.mean():.4f} ± {scores.std():.4f}")

    return results



# --------------------------------------
# 4. Pipeline principal
# --------------------------------------

def pipeline_global(
    df=None,
    csv_path="billets.csv",
    test_size=0.2,
    random_state=42,
    use_scaler=True,
    modele_final='logreg'  
):
    # 1. Chargement des données si non fourni
    if df is None:
        df = charger_donnees(csv_path)
    
    # 2. Préparation des données
    X, y = preparer_donnees(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    # 3. Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Modèles supervisés
    logreg = entrainer_modele(LogisticRegression(), X_train_scaled, y_train)
    knn = entrainer_modele(KNeighborsClassifier(n_neighbors=12), X_train_scaled, y_train)
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
    if modele_final == 'logreg':
        best_model = logreg
    elif modele_final == 'knn':
        best_model = knn
    elif modele_final == 'rf':
        best_model = rf
    else:
        raise ValueError("Modèle final non reconnu. Utilisez 'logreg', 'knn' ou 'rf'.")
    
    return best_model, scaler, X_train_scaled, y_train





def tester_k_meilleurs_voisins(X_train, y_train, max_k=20):
    scores = []
    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        scores.append(cv_scores.mean())
    
    # Plot des résultats
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_k + 1), scores, marker='o')
    plt.title("Performance du KNN en fonction de n_neighbors")
    plt.xlabel("Nombre de voisins (k)")
    plt.ylabel("Accuracy moyenne (validation croisée)")
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()
    
    # Meilleur k
    best_k = scores.index(max(scores)) + 1
    print(f"Meilleur k trouvé : {best_k} avec une accuracy de {max(scores):.4f}")
    return best_k, scores

def verifier_linearite(df):
    features = ['length', 'margin_up', 'height_right']
    for feature in features:
        sns.scatterplot(x=df[feature], y=df['margin_low'])
        plt.title(f"margin_low vs {feature}")
        plt.xlabel(feature)
        plt.ylabel("margin_low")
        plt.show()


