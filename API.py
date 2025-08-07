

import fastapi
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
# === Load model ===
model = joblib.load("Best_risk_model.pkl")

# === Load risk score dictionaries ===
def load_risk(name):
    with open(f"risk_scores/{name}.pkl", "rb") as f:
         return joblib.load(f"risk_scores/{name}.pkl")
risk_scores = {
    "mois": load_risk("mois"),
    "article": load_risk("article"),
    "produit": load_risk("produit"),
    "origine": load_risk("origine"),
    "provenance": load_risk("provenance"),
    "bureau": load_risk("bureau"),
    "commissionnaire": load_risk("commissionnaire"),
    "regime": load_risk("regime"),
    "importateur": load_risk("importateur")
}
# === Load corresponding encoded dictionaries ===
origine_encoded_dict = joblib.load(open("encoded/origine.pkl", "rb"))
provenance_encoded_dict = joblib.load(open("encoded/provenance.pkl", "rb"))
importateur_encoded_dict = joblib.load(open("encoded/importateur.pkl", "rb"))
commissionnaire_encoded_dict = joblib.load(open("encoded/commissionnaire.pkl", "rb"))
regime_encoded_dict = joblib.load(open("encoded/regime.pkl", "rb"))

# === Class needed for the prediction ===
class Marchandise(BaseModel):
    mois: int
    article: int
    produit: int
    origine: str
    provenance: str
    importateur: str
    bureau: str
    commissionnaire: str
    regime: int
    transaction_repetitivite: float

@app.get("/")
def get_predict_circuit_status():
    return  "Prediction endpoint is ready. Please use POST method to make a prediction."

@app.post("/predict-circuit")
def predict_risk(m:Marchandise):
    #warning initialisation
    warnings = []
    # Retrieve internal risk scores
    def get_score(dic, key, name):
        if key ==1777448484: # Default value for unknown keys
            return 0.5
        else:    
            if key not in dic:
                warnings.append(
                    f"⚠️ Key '{key}' not found in {name}_risk_score dictionary. Default score 0.5 was used. Prediction might be less accurate."
                )
                return 0.5
            return dic[key]
    def encode_value(value, encoding_dict, field_name):
        try:
            return encoding_dict[value]
        except KeyError:
            warnings.append(f"⚠️'{value}' not found in '{field_name} ' database,replaced with error value 1777448484, prediction might be less accurate.")
            return 1777448484
    def encode_bureau(bureau_str):
        """
        Converts a bureau code like '18N' to an int like 1814.
        On failure, appends a warning and returns None.
        """
        try:
            if not isinstance(bureau_str, str) or len(bureau_str) != 3:
                raise ValueError()

            num_part = bureau_str[:2]
            letter = bureau_str[2].upper()

            if not num_part.isdigit() or not letter.isalpha():
                raise ValueError()

            letter_index = ord(letter) - ord('A') + 1
            return int(num_part + str(letter_index))
        except Exception:
            warnings.append(
                f"⚠️'{bureau_str}' not found in bureau database,replaced with error value 1777448484, prediction might be less accurate."
            )
            return 1777448484
    origine_encoded=encode_value(m.origine,origine_encoded_dict,"origine_encoded")
    provenance_encoded=encode_value(m.provenance,provenance_encoded_dict,"provenance_encoded")
    bureau_encoded=encode_bureau(m.bureau)
    commissionnaire_encoded=encode_value(m.commissionnaire,commissionnaire_encoded_dict,"commissionnaire_encoded")
    regime_encoded=encode_value(m.regime,regime_encoded_dict,"regime_encoded")
    importateur_encoded=encode_value(m.importateur,importateur_encoded_dict,"importateur_encoded")
    try:
        input_vector = {
            "mois":m.mois,
            "article":m.article,
            "produit":m.produit,
            "origine_encoded":origine_encoded,
            "provenance_encoded":provenance_encoded,
            "importateur_encoded":importateur_encoded,
            "bureau_encoded":bureau_encoded,
            "commissionnaire_encoded":commissionnaire_encoded,
            "regime_encoded":regime_encoded,
            "pays_correspondance":int(m.origine==m.provenance),#if value not equall pays_correspondance 0 else 1 
            "transaction_repetitivite":m.transaction_repetitivite,
            "mois_risk_score":get_score(risk_scores["mois"], m.mois, "mois"),
            "article_risk_score":get_score(risk_scores["article"], m.article, "article"),
            "produit_risk_score":get_score(risk_scores["produit"], m.produit, "produit"),
            "origine_risk_score":get_score(risk_scores["origine"], origine_encoded, "origine"),
            "provenance_risk_score":get_score(risk_scores["provenance"], provenance_encoded, "provenance"),
            "bureau_risk_score":get_score(risk_scores["bureau"], bureau_encoded, "bureau"),
            "commissionnaire_risk_score":get_score(risk_scores["commissionnaire"], commissionnaire_encoded, "commissionnaire"),
            "regime_risk_score":get_score(risk_scores["regime"], regime_encoded, "regime"),
            "importateur_risk_score":get_score(risk_scores["importateur"], importateur_encoded, "importateur")
        }
    except Exception as e :
        raise HTTPException(status_code=400, detail=str(e))


    X = pd.DataFrame([input_vector])
    prediction = model.predict(X)
    label_map = {0: "Circuit Vert", 1: "Circuit Orange", 2: "Circuit Rouge"}
    prediction_int = int(prediction[0])  # Convert from np.int64 to native int
    label = label_map.get(prediction_int, "Unknown")

    return {
    "model":model.named_steps['clf'].__class__.__name__,
    "PCA": 'pca' in model.named_steps,
    "prediction": prediction_int,
    "label": label,
    "warnings": warnings if warnings else None
}
