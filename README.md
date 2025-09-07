# 🛃 Système IA - Prédiction Circuits Douaniers

Application Streamlit pour la prédiction automatisée des circuits de contrôle douanier (Vert, Orange, Rouge) utilisant l'intelligence artificielle.

## 🚀 Installation

```bash
# Cloner le projet
git clone https://bitbucket.org/jasmineconseil-fr/douane-ia.git
cd douane-ia

# Créer l'environnement virtuel
python -m venv douane_ia_env
source douane_ia_env/bin/activate  # Linux/Mac
# douane_ia_env\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 📁 Structure des Fichiers

```
douane_project/
├── streamlit_app.py           # Application principale
├── API.py                     # API FastAPI pour prédictions
├── run_streamlit.py           # Script de démarrage
├── Final_resultas_risques.csv # Données d'entraînement
├── resultats_risque.csv       # Mappings des vraies valeurs
└── Best_risk_model.pkl        # Modèle ML entraîné
```

## 🎯 Utilisation

### 1- Démarrage automatique rapide
```bash
python run_streamlit.py
```

### 2- Démarrage contrôlée manuel
```bash
# Lancer l'API (terminal 1)
uvicorn API:app --reload --port 8000 &

# Lancer Streamlit (terminal 2)  
streamlit run streamlit_app.py --server.port 8501
```

## 📊 Fonctionnalités

- **🏠 Dashboard Principal** : Vue d'ensemble avec métriques clés
- **📊 Exploration des Données** : Analyses statistiques et visualisations
- **🎯 Test de Prédiction** : Interface pour tester de nouvelles déclarations
- **⚠️ Analyse des Risques** : Identification des variables critiques
- **📈 Métriques de Performance** : Évaluation du modèle

## 🌐 Accès

- **Streamlit** : http://localhost:8501
- **API FastAPI** : http://localhost:8000
- **Documentation API** : http://localhost:8000/docs

## 🔧 Technologies

- **Frontend** : Streamlit, Plotly
- **Backend** : FastAPI, scikit-learn
- **Data** : Pandas, NumPy
- **ML** : Random Forest, encodage LabelEncoder

## 👥 Développement

Développé dans le cadre du projet de modernisation du système douanier sénégalais pour optimiser les contrôles et réduire les délais de traitement.

---
*Dernière mise à jour : 2025*