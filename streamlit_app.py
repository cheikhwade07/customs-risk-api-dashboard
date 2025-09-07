## 2e version streamlit - version actuelle - 5 Aout 2025
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🛃 Aide à la Décision Douanière - IA",
    page_icon="🛃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
     body { background-color: white; color: black; }
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    .risk-high { border-left-color: #ef4444 !important; }
    .risk-medium { border-left-color: #f59e0b !important; }
    .risk-low { border-left-color: #10b981 !important; }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("""
<div class="main-header">
    <h1>🛃 Système d'Aide à la Décision Douanière</h1>
    <p>Modélisation IA pour la prédiction des circuits de contrôle</p>
</div>
""", unsafe_allow_html=True)

# Fonctions utilitaires
@st.cache_data
def load_data():
    """Charge les données depuis le CSV"""
    try:
        df = pd.read_csv("Final_resultas_risques.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Fichier 'Final_resultas_risques.csv' non trouvé. Assurez-vous qu'il est dans le bon répertoire.")
        return None

@st.cache_data
def load_mapping_data():
    """Charge les données de mapping depuis resultats_risques.csv"""
    try:
        df = pd.read_csv("resultats_risques.csv")
        return df
    except FileNotFoundError:
        st.warning("⚠️ Fichier 'resultats_risques.csv' non trouvé. Utilisation de valeurs par défaut pour les mappings.")
        return None

@st.cache_data
def get_mappings():
    """Crée des mappings réels basés sur les données ou utilise des valeurs par défaut"""
    
    mapping_df = load_mapping_data()
    
    if mapping_df is not None:
        # Créer des mappings à partir des vraies données
        mappings = {}
        
        # Mapping pays origine
        if 'origine' in mapping_df.columns and 'origine_encoded' in mapping_df.columns:
            origine_map = mapping_df[['origine', 'origine_encoded']].drop_duplicates()
            mappings['origine'] = dict(zip(origine_map['origine'], origine_map['origine_encoded']))
        
        # Mapping pays provenance
        if 'provenance' in mapping_df.columns and 'provenance_encoded' in mapping_df.columns:
            provenance_map = mapping_df[['provenance', 'provenance_encoded']].drop_duplicates()
            mappings['provenance'] = dict(zip(provenance_map['provenance'], provenance_map['provenance_encoded']))
        
        # Mapping importateurs
        if 'importateur' in mapping_df.columns and 'importateur_encoded' in mapping_df.columns:
            importateur_map = mapping_df[['importateur', 'importateur_encoded']].drop_duplicates()
            mappings['importateur'] = dict(zip(importateur_map['importateur'], importateur_map['importateur_encoded']))
        
        # Mapping commissionnaires
        if 'commissionnaire' in mapping_df.columns and 'commissionnaire_encoded' in mapping_df.columns:
            commissionnaire_map = mapping_df[['commissionnaire', 'commissionnaire_encoded']].drop_duplicates()
            mappings['commissionnaire'] = dict(zip(commissionnaire_map['commissionnaire'], commissionnaire_map['commissionnaire_encoded']))
        
        # Mapping régimes
        if 'regime' in mapping_df.columns and 'regime_encoded' in mapping_df.columns:
            regime_map = mapping_df[['regime', 'regime_encoded']].drop_duplicates()
            mappings['regime'] = dict(zip(regime_map['regime'], regime_map['regime_encoded']))
        
        # Mapping bureaux (si disponible)
        if 'bureau' in mapping_df.columns:
            bureau_list = mapping_df['bureau'].dropna().unique()
            mappings['bureau'] = {bureau: bureau for bureau in bureau_list}
        
        # Mapping produits (utiliser libelle_produit si disponible)
        if 'libelle_produit' in mapping_df.columns and 'produit' in mapping_df.columns:
            produit_map = mapping_df[['libelle_produit', 'produit']].dropna().drop_duplicates()
            # Limiter à 50 produits les plus fréquents pour éviter une liste trop longue
            produit_counts = mapping_df['produit'].value_counts().head(50)
            top_produits = produit_map[produit_map['produit'].isin(produit_counts.index)]
            mappings['produit'] = dict(zip(top_produits['libelle_produit'], top_produits['produit']))
        
        return mappings
    
    else:
        # Mappings par défaut si le fichier n'est pas disponible
        return {
            'origine': {
                "Sénégal": 96, "France": 35, "Chine": 21, "Allemagne": 3, 
                "Japon": 53, "Inde": 46, "États-Unis": 106, "Royaume-Uni": 93,
                "Maroc": 64, "Algérie": 15, "Tunisie": 18, "Espagne": 33,
                "Italie": 52, "Belgique": 11, "Pays-Bas": 47, "Brésil": 61,
                "Corée du Sud": 75, "Turquie": 105, "Russie": 91, "Canada": 95
            },
            'provenance': {
                "Sénégal": 93, "France": 35, "Chine": 21, "Allemagne": 3, 
                "Japon": 53, "Inde": 48, "États-Unis": 106, "Royaume-Uni": 93,
                "Maroc": 65, "Algérie": 15, "Tunisie": 18, "Espagne": 33,
                "Italie": 54, "Belgique": 11, "Pays-Bas": 47, "Brésil": 61,
                "Corée du Sud": 75, "Turquie": 100, "Russie": 91, "Canada": 95
            },
            'importateur': {
                "SONACOS SA": 178, "SENELEC": 401, "SUNEOR": 502, "PATISEN": 1084,
                "GRANDS MOULINS": 1108, "SENEGAL EXPORT": 1247, "CORIS BANK": 1399,
                "ECOBANK": 2367, "BOLLORÉ LOGISTICS": 3071, "DHL SÉNÉGAL": 1501
            },
            'commissionnaire': {
                "DOUANES SERVICES": 21, "RAPID CLEARING": 28, "TRANSIT PLUS": 38,
                "WEST AFRICA TRANSIT": 78, "SENEGAL CUSTOMS": 83, "EASY TRANSIT": 160,
                "QUICK CLEARING": 181, "PROFESSIONAL TRANSIT": 198
            },
            'regime': {
                "Admission temporaire": 1450, "Mise à la consommation": 4000,
                "Transit": 8797, "Entrepôt": 6784, "Perfectionnement actif": 9268
            },
            'bureau': {
                "18N": "18N", "10S": "10S", "13L": "13L", "19M": "19M", "20P": "20P"
            },
            'produit': {
                "Véhicules automobiles": 8703332000, "Autres véhicules": 8704212000,
                "Huiles de pétrole": 2709000000, "Téléphones": 8517120000,
                "Ordinateurs": 8471300000
            }
        }

def get_risk_category_name(category):
    """Convertit les catégories numériques en noms"""
    mapping = {0: "Vert (Risque Faible)", 1: "Orange (Risque Modéré)", 2: "Rouge (Très Susceptible de Fraude)"}
    return mapping.get(category, "Inconnu")

def predict_via_api(data, api_url="http://localhost:8000/predict-circuit"):
    """Appelle l'API pour faire une prédiction"""
    try:
        response = requests.post(api_url, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Erreur API: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"Impossible de contacter l'API: {str(e)}"}

# Chargement des mappings
mappings = get_mappings()

# Sidebar pour navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.selectbox(
    "Choisir une section",
    ["🏠 Dashboard Principal", "📊 Exploration des Données", "🎯 Test de Prédiction", "⚠️ Analyse des Risques", "📈 Métriques de Performance"]
)

# Chargement des données
df = load_mapping_data()

if df is not None:
    
    # Définir risk_cols une seule fois au début
    risk_cols = [col for col in df.columns if 'risk_score' in col and col != 'risk_score']
    
    # === PAGE 1: DASHBOARD PRINCIPAL ===
    if page == "🏠 Dashboard Principal":
        st.header("📋 Vue d'ensemble des données")
        
        # Métriques générales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📦 Total Déclarations</h3>
                <h2>{:,}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            # Adapter selon le format des données
            if 'risk_category' in df.columns:
                if df['risk_category'].dtype == 'object':
                    # Format texte
                    risk_counts = df['risk_category'].value_counts()
                    vert_count = risk_counts.get("Vert (Risque Faible)", 0)
                else:
                    # Format numérique
                    vert_count = len(df[df['risk_category'] == 0])
            else:
                vert_count = 0
                
            st.markdown("""
            <div class="metric-card risk-low">
                <h3>🟢 Risque Faible</h3>
                <h2>{:,} ({:.1f}%)</h2>
            </div>
            """.format(vert_count, vert_count/len(df)*100 if len(df) > 0 else 0), unsafe_allow_html=True)
        
        with col3:
            if 'risk_category' in df.columns:
                if df['risk_category'].dtype == 'object':
                    orange_count = risk_counts.get("Orange (Risque Modéré)", 0)
                else:
                    orange_count = len(df[df['risk_category'] == 1])
            else:
                orange_count = 0
                
            st.markdown("""
            <div class="metric-card risk-medium">
                <h3>🟡 Risque Modéré</h3>
                <h2>{:,} ({:.1f}%)</h2>
            </div>
            """.format(orange_count, orange_count/len(df)*100 if len(df) > 0 else 0), unsafe_allow_html=True)
        
        with col4:
            if 'risk_category' in df.columns:
                if df['risk_category'].dtype == 'object':
                    rouge_count = risk_counts.get("Rouge (Très Susceptible de Fraude)", 0)
                else:
                    rouge_count = len(df[df['risk_category'] == 2])
            else:
                rouge_count = 0
                
            st.markdown("""
            <div class="metric-card risk-high">
                <h3>🔴 Risque Élevé</h3>
                <h2>{:,} ({:.1f}%)</h2>
            </div>
            """.format(rouge_count, rouge_count/len(df)*100 if len(df) > 0 else 0), unsafe_allow_html=True)
        
        # Distribution des catégories de risque
        st.subheader("📊 Distribution des Catégories de Risque")
        if 'risk_category' in df.columns:
            if df['risk_category'].dtype == 'object':
                risk_counts = df['risk_category'].value_counts()
            else:
                # Convertir les codes numériques en noms
                risk_counts = df['risk_category'].map(get_risk_category_name).value_counts()
            
            # Ensure index is string and trimmed
        risk_counts.index = risk_counts.index.astype(str).str.strip()

        # Now create the pie chart
        fig_pie = px.pie(
            values=risk_counts.values, 
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={
                "Vert (Risque Faible)": "#22F6AF",
                "Orange (Risque Modéré)": "#e7a711", 
                "Rouge (Très Susceptible de Fraude)": "#DC1616"
            },
            category_orders={
                "names": [
                    "Vert (Risque Faible)", 
                    "Orange (Risque Modéré)", 
                    "Rouge (Très Susceptible de Fraude)"
                ]
            },
            title="Répartition des Déclarations par Niveau de Risque"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        # Aperçu des données
        st.subheader("🔍 Aperçu des Données")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Informations sur les colonnes
        st.subheader("📋 Description des Variables")
        col_info = {
            'mois': 'Mois de la déclaration',
            'article': 'Numéro d\'article',
            'produit': 'Code produit',
            'origine_encoded': 'Pays d\'origine (encodé)',
            'provenance_encoded': 'Pays de provenance (encodé)',
            'importateur_encoded': 'Importateur (encodé)',
            'bureau_encoded': 'Bureau de douane (encodé)',
            'commissionnaire_encoded': 'Commissionnaire (encodé)',
            'regime_encoded': 'Régime douanier (encodé)',
            'pays_correspondance': 'Correspondance origine/provenance (1=oui, 0=non)',
            'transaction_repetitivite': 'Score de répétitivité des transactions',
            'risk_score': 'Score de risque calculé (0-1)',
            'risk_category': 'Catégorie de risque prédite'
        }
        available_cols = {k: v for k, v in col_info.items() if k in df.columns}
        info_df = pd.DataFrame(list(available_cols.items()), columns=['Variable', 'Description'])
        st.dataframe(info_df, use_container_width=True)

    # === PAGE 2: EXPLORATION DES DONNÉES ===
    elif page == "📊 Exploration des Données":
        st.header("📊 Analyse Exploratoire des Données")
        
        # Distribution des scores de risque
        st.subheader("📈 Distribution des Scores de Risque")
        if 'risk_score' in df.columns and 'risk_category' in df.columns:
            # Adapter selon le format des risk_category
            if df['risk_category'].dtype == 'object':
                color_col = 'risk_category'
            else:
                df_temp = df.copy()
                df_temp['risk_category_name'] = df_temp['risk_category'].map(get_risk_category_name)
                color_col = 'risk_category_name'
                df = df_temp  # Utiliser la version temporaire
            
            fig_hist = px.histogram(
                df, x='risk_score', color=color_col,
                nbins=50,
                title="Distribution des Scores de Risque par Catégorie",
                color_discrete_map={
                    "Vert (Risque Faible)": "#10b981",
                    "Orange (Risque Modéré)": "#f59e0b", 
                    "Rouge (Très Susceptible de Fraude)": "#ef4444"
                }
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Analyse par bureau (en utilisant bureau_encoded si bureau n'existe pas)
        st.subheader("🏢 Analyse par Bureau de Douane")
        bureau_col = 'bureau' if 'bureau' in df.columns else 'bureau_encoded'
        
        if bureau_col in df.columns and 'risk_category' in df.columns:
            bureau_risk = df.groupby([bureau_col, color_col]).size().reset_index(name='count')
            fig_bureau = px.bar(
                bureau_risk, x=bureau_col, y='count', color=color_col,
                title="Distribution des Risques par Bureau",
                color_discrete_map={
                    "Vert (Risque Faible)": "#10b981",
                    "Orange (Risque Modéré)": "#f59e0b", 
                    "Rouge (Très Susceptible de Fraude)": "#ef4444"
                }
            )
            st.plotly_chart(fig_bureau, use_container_width=True)
        
        # Analyse par origine (utiliser origine_encoded si origine n'existe pas)
        st.subheader("🌍 Analyse par Pays d'Origine")
        origine_col = 'origine' if 'origine' in df.columns else 'origine_encoded'
        
        if origine_col in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                origine_counts = df[origine_col].value_counts().head(10)
                fig_origine = px.bar(
                    x=origine_counts.values, y=origine_counts.index,
                    orientation='h',
                    title="Top 10 des Pays d'Origine",
                    labels={'x': 'Nombre de déclarations', 'y': 'Pays'}
                )
                st.plotly_chart(fig_origine, use_container_width=True)
            
            with col2:
                # Risque moyen par origine
                if 'risk_score' in df.columns:
                    risk_by_origine = df.groupby(origine_col)['risk_score'].mean().sort_values(ascending=False).head(10)
                    fig_risk_origine = px.bar(
                        x=risk_by_origine.values, y=risk_by_origine.index,
                        orientation='h',
                        title="Score de Risque Moyen par Origine",
                        labels={'x': 'Score de risque moyen', 'y': 'Pays'},
                        color=risk_by_origine.values,
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_risk_origine, use_container_width=True)
        
        # Évolution temporelle (utiliser mois si date n'existe pas)
        st.subheader("📅 Évolution Temporelle")
        if 'date' in df.columns:
            df_temp = df.copy()
            df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
            df_temp['mois'] = df_temp['date'].dt.to_period('M')
            temporal_analysis = df_temp.groupby(['mois', color_col]).size().reset_index(name='count')
            temporal_analysis['mois_str'] = temporal_analysis['mois'].astype(str)
            x_col = 'mois_str'
        elif 'mois' in df.columns:
            temporal_analysis = df.groupby(['mois', color_col]).size().reset_index(name='count')
            x_col = 'mois'
        else:
            st.info("Pas de données temporelles disponibles")
            temporal_analysis = None
        
        if temporal_analysis is not None:
            fig_temporal = px.line(
                temporal_analysis, x=x_col, y='count', color=color_col,
                title="Évolution du Nombre de Déclarations par Catégorie de Risque",
                color_discrete_map={
                    "Vert (Risque Faible)": "#10b981",
                    "Orange (Risque Modéré)": "#f59e0b", 
                    "Rouge (Très Susceptible de Fraude)": "#ef4444"
                }
            )
            st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Matrice de corrélation
        st.subheader("🔗 Matrice de Corrélation")
        numeric_cols = ['risk_score', 'transaction_repetitivite', 'pays_correspondance']
        
        # Ajouter les colonnes encodées disponibles
        encoded_cols = [col for col in df.columns if '_encoded' in col]
        numeric_cols.extend(encoded_cols)
        
        # Ajouter d'autres colonnes numériques si disponibles
        other_numeric = ['mois', 'article', 'produit', 'regime']
        for col in other_numeric:
            if col in df.columns:
                numeric_cols.append(col)
        
        # Garder seulement les colonnes qui existent
        available_numeric = [col for col in numeric_cols if col in df.columns]
        
        if len(available_numeric) > 1:
            corr_matrix = df[available_numeric].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Matrice de Corrélation des Variables Numériques",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    # === PAGE 3: TEST DE PRÉDICTION ===
    elif page == "🎯 Test de Prédiction":
        st.header("🎯 Test de Prédiction")
        
        st.info("💡 Cette page vous permet de tester l'API de prédiction avec de nouvelles données.")
        
        # URL de l'API
        api_url = st.text_input(
            "URL de l'API:", 
            value="http://localhost:8000/predict-circuit"
        )
        
        # Formulaire de saisie avec de vraies valeurs
        st.subheader("📝 Saisie des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mois = st.number_input("Mois:", min_value=1, max_value=12, value=1)
            article = st.number_input("Article:", min_value=0, value=100000)
            
            # Selectbox pour produit si mapping disponible
            if 'produit' in mappings and mappings['produit']:
                produit_names = list(mappings['produit'].keys())
                selected_produit_name = st.selectbox("Produit:", produit_names)
                produit = mappings['produit'][selected_produit_name]
                st.write(f"Code produit: {produit}")
            else:
                produit = st.number_input("Produit:", min_value=0, value=200000)
            
            # Selectbox pour origine
            if 'origine' in mappings and mappings['origine']:
                origine_names = list(mappings['origine'].keys())
                selected_origine = st.selectbox("Pays d'Origine:", origine_names)
                origine = selected_origine
            else:
                origine = st.text_input("Origine:", value="France")
            
            # Selectbox pour provenance
            if 'provenance' in mappings and mappings['provenance']:
                provenance_names = list(mappings['provenance'].keys())
                selected_provenance = st.selectbox("Pays de Provenance:", provenance_names)
                provenance = selected_provenance
            else:
                provenance = st.text_input("Provenance:", value="France")
        
        with col2:
            # Selectbox pour importateur
            if 'importateur' in mappings and mappings['importateur']:
                importateur_names = list(mappings['importateur'].keys())
                selected_importateur = st.selectbox("Importateur:", importateur_names)
                importateur = selected_importateur
            else:
                importateur = st.text_input("Importateur:", value="IMP001")
            
            # Selectbox pour bureau
            if 'bureau' in mappings and mappings['bureau']:
                bureau_names = list(mappings['bureau'].keys())
                selected_bureau = st.selectbox("Bureau:", bureau_names)
                bureau = selected_bureau
            else:
                bureau = st.text_input("Bureau:", value="18N")
            
            # Selectbox pour commissionnaire
            if 'commissionnaire' in mappings and mappings['commissionnaire']:
                commissionnaire_names = list(mappings['commissionnaire'].keys())
                selected_commissionnaire = st.selectbox("Commissionnaire:", commissionnaire_names)
                commissionnaire = selected_commissionnaire
            else:
                commissionnaire = st.text_input("Commissionnaire:", value="COM001")
            
            # Selectbox pour régime
            if 'regime' in mappings and mappings['regime']:
                regime_names = list(mappings['regime'].keys())
                selected_regime_name = st.selectbox("Régime:", regime_names)
                regime = selected_regime_name 
        
            else:
                regime = st.number_input("Régime:", min_value=0, value=4000)
            
            transaction_repetitivite = st.number_input(
                "Répétitivité Transaction:", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5,
                step=0.1
            )
        
        # Bouton de prédiction
        if st.button("🔮 Faire une Prédiction", type="primary"):
            # Préparer les données
            prediction_data ={
                "mois": mois,
                "article": article,
                "produit": produit,
                "origine": origine,
                "provenance": provenance,
                "importateur": importateur,
                "bureau": bureau,
                "commissionnaire": commissionnaire,
                "regime": regime,
                "transaction_repetitivite": transaction_repetitivite
            }
            
            # Afficher les données envoyées
            st.write("**Données envoyées à l'API:**")
            st.json(prediction_data)
            
            # Appeler l'API
            with st.spinner("⏳ Prédiction en cours..."):
                result = predict_via_api(prediction_data, api_url)
            
            # Afficher le résultat
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                st.success("✅ Prédiction réussie!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Résultat de la Prédiction")
                    
                    # Couleur selon le circuit
                    circuit_colors = {
                        "Circuit Vert": "#10b981",
                        "Circuit Orange": "#f59e0b", 
                        "Circuit Rouge": "#ef4444"
                    }
                    
                    label = result.get("label", "Inconnu")
                    color = circuit_colors.get(label, "#6b7280")
                    
                    st.markdown(f"""
                    <div style="background-color: {color}; color: white; padding: 2rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
                        <h2>Circuit Prédit: {label}</h2>
                        <h3>Code: {result.get('prediction', 'N/A')}</h3>
                        <p>Modèle: {result.get('model', 'N/A')}</p>
                        <p>PCA: {'Oui' if result.get('PCA', False) else 'Non'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("⚠️ Avertissements")
                    warnings = result.get("warnings")
                    if warnings:
                        for warning in warnings:
                            st.warning(warning)
                    else:
                        st.success("Aucun avertissement - Toutes les variables sont connues du modèle")

    # === PAGE 4: ANALYSE DES RISQUES ===
    elif page == "⚠️ Analyse des Risques":
        st.header("⚠️ Analyse des Risques")
        
        # Adapter selon le format des risk_category
        if 'risk_category' in df.columns:
            if df['risk_category'].dtype == 'object':
                color_col = 'risk_category'
                risk_categories = ["Vert (Risque Faible)", "Orange (Risque Modéré)", "Rouge (Très Susceptible de Fraude)"]
            else:
                df_temp = df.copy()
                df_temp['risk_category_name'] = df_temp['risk_category'].map(get_risk_category_name)
                color_col = 'risk_category_name'
                risk_categories = ["Vert (Risque Faible)", "Orange (Risque Modéré)", "Rouge (Très Susceptible de Fraude)"]
                df = df_temp
        
        # Analyse par circuit
        st.subheader("📊 Analyse par Circuit de Risque")
        
        # Tableau de synthèse
        if 'risk_score' in df.columns and 'transaction_repetitivite' in df.columns:
            circuit_summary = []
            for i, category in enumerate(risk_categories):
                if df['risk_category'].dtype == 'object':
                    subset = df[df['risk_category'] == category]
                else:
                    subset = df[df['risk_category'] == i]
                
                circuit_summary.append({
                    'Circuit': category,
                    'Nombre': len(subset),
                    'Pourcentage': len(subset)/len(df)*100,
                    'Score Moyen': subset['risk_score'].mean() if len(subset) > 0 else 0,
                    'Répétitivité Moyenne': subset['transaction_repetitivite'].mean() if len(subset) > 0 else 0
                })
            
            circuit_summary_df = pd.DataFrame(circuit_summary).round(2)
            st.dataframe(circuit_summary_df)
        
        # Heatmap des risques par variable
        if risk_cols:
            st.subheader("🔥 Heatmap des Scores de Risque")
            
            # Calculer les moyennes par circuit
            if df['risk_category'].dtype == 'object':
                risk_by_circuit = df.groupby('risk_category')[risk_cols].mean()
            else:
                risk_by_circuit = df.groupby('risk_category')[risk_cols].mean()
                risk_by_circuit.index = [get_risk_category_name(i) for i in risk_by_circuit.index]
            
            fig = px.imshow(
                risk_by_circuit.T,
                title="Scores de Risque Moyens par Circuit et Variable",
                labels={'x': 'Circuit', 'y': 'Variable', 'color': 'Score de Risque'},
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top variables de risque
            st.subheader("🔝 Variables les Plus Risquées")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 10 - Scores de Risque Moyens:**")
                top_risks = df[risk_cols].mean().sort_values(ascending=False).head(10)
                for var, score in top_risks.items():
                    st.write(f"• {var}: {score:.3f}")
            
            with col2:
                st.write("**Top 10 - Écart-types (Variabilité):**")
                top_vars = df[risk_cols].std().sort_values(ascending=False).head(10)
                for var, std in top_vars.items():
                    st.write(f"• {var}: {std:.3f}")

    # === PAGE 5: MÉTRIQUES DE PERFORMANCE ===
    elif page == "📈 Métriques de Performance":
        st.header("📈 Métriques de Performance")
        st.info("💡 Cette page présente l'analyse de performance du modèle de détection de risques.")

        model = joblib.load("Best_risk_model.pkl")
        df = load_data()
        df = df.drop(df.columns[0], axis=1)

        # Distribution des scores finaux
        st.subheader("📊 Distribution des Scores Finaux")

        # Créer deux colonnes pour la matrice de confusion et l'importance des variables
        col1, col2 = st.columns(2)

        # Vérifier que les colonnes nécessaires existent
        if 'risk_category' in df.columns:
            # Préparation des données
            X = df.drop(columns=['risk_category', 'risk_score'])
            df['risk_category'] = df['risk_category'] - 1
            y_true = df['risk_category']
            y_pred = model.predict(X)

            # --- Colonne 1 : Matrice de confusion ---
            # Extraire le nom du classifieur final
            model_name = model.steps[-1][1].__class__.__name__

            # Vérifier s’il y a un PCA dans le pipeline
            has_pca = any("pca" in step[0].lower() and "PCA" in step[1].__class__.__name__ for step in model.steps)

            # Construire le titre dynamiquement
            title = f"Matrice de Confusion du {model_name}"
            if has_pca:
                title += " avec PCA"

            with col1:
                st.subheader(title)
                cm = confusion_matrix(y_true, y_pred)
                fig_cm, ax_cm = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Vert", "Orange", "Rouge"])
                disp.plot(ax=ax_cm, cmap='Blues')
                st.pyplot(fig_cm)

            # --- Colonne 2 : Importance des variables ---
            with col2:
                st.subheader("   Top 10 Variables les Plus Impactantes")

                clf = model
                if hasattr(model, "named_steps"):
                    clf = model.named_steps.get("clf") or model.steps[-1][1]

                if hasattr(clf, "feature_importances_"):
                    importances = clf.feature_importances_
                    top10 = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(10)
                    fig_top, ax_top = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=top10.values, y=top10.index, palette="viridis", ax=ax_top)
                    ax_top.set_title("Top 10 Features Importantes")
                    st.pyplot(fig_top)

                else:
                    from sklearn.inspection import permutation_importance

    

                # Compute permutation importance
                result = permutation_importance(
                    model, X, y_true, 
                    n_repeats=10,
                    random_state=42,
                    scoring='f1_macro'  # You can also use 'accuracy'
                )

                top_perm = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False).head(10)

                # Plot
                fig_perm, ax_perm = plt.subplots(figsize=(6, 4))
                sns.barplot(x=top_perm.values, y=top_perm.index, palette="Blues", ax=ax_perm)
                ax_perm.set_title("Top 10 Variables - Importance par Permutation")
                st.pyplot(fig_perm)
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    🛃 <strong>Système d'Aide à la Décision Douanière</strong><br>
    Développé pour optimiser les contrôles douaniers | 
    Dernière mise à jour: {}
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)