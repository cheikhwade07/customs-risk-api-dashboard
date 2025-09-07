#!/usr/bin/env python3
"""
Script de démarrage pour la plateforme Streamlit - Système IA Douane
Usage: python run_streamlit.py
"""

import subprocess
import sys
import os
from pathlib import Path

def check_file_exists(filename):
    """Vérifie si un fichier existe"""
    if not Path(filename).exists():
        print(f"❌ ERREUR: Le fichier '{filename}' est manquant.")
        return False
    return True

def check_requirements():
    """Vérifie que tous les fichiers nécessaires sont présents"""
    required_files = [
        "Final_resultas_risques.csv",
        "API.py",
        "Best_risk_model.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        if not check_file_exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("💡 Fichiers manquants:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n📁 Structure attendue du projet:")
        print("douane_project/")
        print("├── streamlit_app.py")
        print("├── API.py") 
        print("├── Final_resultas_risques.csv")
        print("├── Best_risk_model.pkl")
        print("├── risk_scores/")
        print("│   ├── mois.pkl")
        print("│   ├── article.pkl")
        print("│   └── ...")
        print("└── encoded/")
        print("    ├── origine.pkl")
        print("    ├── provenance.pkl")
        print("    └── ...")
        return False
    
    return True

def install_requirements():
    """Installe les dépendances si nécessaire"""
    try:
        import streamlit
        import plotly
        import pandas
        print("✅ Dépendances déjà installées")
        return True
    except ImportError:
        print("📦 Installation des dépendances...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dépendances installées avec succès")
            return True
        except subprocess.CalledProcessError:
            print("❌ Erreur lors de l'installation des dépendances")
            return False

def start_api_server():
    """Démarre le serveur API en arrière-plan"""
    try:
        print("🚀 Démarrage du serveur API...")
        # Lancer l'API en arrière-plan
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "API:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
        print("✅ Serveur API démarré sur http://localhost:8000")
        return api_process
    except Exception as e:
        print(f"⚠️  Impossible de démarrer l'API: {e}")
        print("💡 Vous devrez la démarrer manuellement avec: uvicorn API:app --reload")
        return None

def start_streamlit():
    """Démarre l'application Streamlit"""
    try:
        print("🌟 Démarrage de l'application Streamlit...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application arrêtée par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur lors du démarrage de Streamlit: {e}")

def main():
    """Fonction principale"""
    print("🚢 Système IA - Prédiction Circuits Douaniers")
    print("=" * 50)
    
    # Vérifier les fichiers requis
    if not check_requirements():
        print("\n💡 Assurez-vous d'avoir tous les fichiers nécessaires avant de continuer.")
        return
    
    # Installer les dépendances
    if not install_requirements():
        return
    
    print("\n🎯 Choisissez une option:")
    print("1. Lancer Streamlit seulement")
    print("2. Lancer API + Streamlit")
    print("3. Lancer API seulement")
    print("4. Quitter")
    
    choice = input("\nVotre choix (1-4): ").strip()
    
    if choice == "1":
        start_streamlit()
    elif choice == "2":
        api_process = start_api_server()
        if api_process:
            try:
                start_streamlit()
            finally:
                print("\n🛑 Arrêt du serveur API...")
                api_process.terminate()
    elif choice == "3":
        start_api_server()
        input("\nAppuyez sur Entrée pour arrêter le serveur API...")
    elif choice == "4":
        print("👋 Au revoir!")
    else:
        print("❌ Choix invalide")

if __name__ == "__main__":
    main()