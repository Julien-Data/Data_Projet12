# Détection de faux billets – ONCFM
Projet de Machine Learning pour la prédiction de billets vrais ou faux

## 1. Contexte
L’ONCFM souhaite lutter contre la contrefaçon en mettant à disposition des équipes une application de Machine Learning capable d’identifier les faux billets en euros à partir de mesures physiques (longueur, hauteur, largeur, etc.).  

Le projet repose sur un dataset de **1500 billets scannés** :  
- 1000 vrais billets  
- 500 faux billets  

L’objectif est de développer un modèle prédictif et une application fonctionnelle utilisable par les équipes sur le terrain.

## 2. Données utilisées
- `billets.csv` : mesures physiques des billets (longueur, largeur, hauteur, poids, etc.)  
- Cahier des charges détaillant les contraintes et objectifs du projet  

## 3. Méthodologie
1. **Analyse exploratoire et prétraitement**
   - Nettoyage des données  
   - Vérification des valeurs manquantes et des outliers  
   - Normalisation / standardisation si nécessaire  

2. **Modélisation Machine Learning**
   - Algorithmes testés :
     - K-means  
     - Régression logistique  
     - K-Nearest Neighbors (KNN)  
     - Random Forest  
   - Évaluation des performances (accuracy, précision, rappel, F1-score)  
   - Sélection du modèle final en fonction des résultats  

3. **Développement de l’application**
   - Script Python indépendant (`app_prediction.py`) permettant de prédire la nature d’un billet à partir de ses caractéristiques  

4. **Documentation et résultats**
   - Notebook Jupyter détaillant :
     - Analyse exploratoire  
     - Prétraitements  
     - Comparaison des algorithmes  
     - Justification du choix du modèle final  
     - Résultats et visualisations  

## 4. Instructions pour lancer le projet
1. Installer Python >= 3.10 et VS Code  
2. Installer l’extension Jupyter pour VS Code  
3. Installer les librairies nécessaires :
   ```bash
   pip install -r requirements.txt

