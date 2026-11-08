from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin

app = FastAPI(
    title="API de Predicción de Abandono (Churn)",
    description="Servicio Cloud de IA con XGBoost - Proyecto Computacion en la Nube para IA"
)

# Definición de la clase personalizada requerida por el modelo serializado
class TelcoFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_out = X.copy()
        X_out['TotalCharges'] = pd.to_numeric(X_out['TotalCharges'], errors='coerce').fillna(0.0)
        X_out['ChargesRatio'] = X_out['MonthlyCharges'] / (X_out['tenure'] + 1)
        if 'customerID' in X_out.columns:
            X_out = X_out.drop(columns=['customerID'])
        return X_out

# Cargar el pipeline de procesamiento e inferencia
MODEL_PATH = "churn_pipeline_v1.joblib"
modelo = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

class ClienteData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

@app.get("/")
def health_check():
    return {"status": "Servicio Cloud Activo", "modelo": "XGBoost Churn Pipeline"}

@app.post("/predecir")
def predecir_abandono(cliente: ClienteData):
    if not modelo:
        return {"error": "El artefacto del modelo no se encontró en el servidor."}
    
    # Transformar el JSON entrante al formato que espera tu pipeline
    df_cliente = pd.DataFrame([cliente.model_dump()])
    
    # Inferencia (el pipeline ya incluye la limpieza y variables creadas)
    probabilidad = modelo.predict_proba(df_cliente)[0][1]
    diagnostico = "ALERTA: Alto Riesgo de Abandono" if probabilidad >= 0.5 else "Bajo Riesgo (Retenido)"
    
    return {
        "probabilidad_churn": round(float(probabilidad), 4),
        "accion_recomendada": diagnostico
    }
