from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd
import requests
import os

# -------------------------
# Load trained objects
# -------------------------
model = joblib.load("malnutrition_model.pkl")
imputer = joblib.load("imputer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Malnutrition Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Input schema
# -------------------------
class InputData(BaseModel):
    age_months: int
    gender: str
    height_cm: float
    weight_kg: float
    muac_cm: float

# -------------------------
# Agentic AI function (OPTIONAL)
# -------------------------
def generate_nutrition_suggestions(prediction: str) -> str:
    API_KEY =  os.getenv("GROQ_API_KEY")
# you already have it
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a compassionate nutrition assistant.
The predicted nutritional status is: {prediction}.
Give realistic, caring, and practical dietary suggestions.
Do not give medical prescriptions.
Encourage consulting healthcare professionals when needed.
"""

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a caring nutrition assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception:
        return "AI suggestions unavailable at the moment."

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        # ---- derive internal features ----
        height_m = data.height_cm / 100
        bmi = data.weight_kg / (height_m ** 2)

        sex_enc = 1 if data.gender.lower() == "male" else 0

        if data.muac_cm < 11.5:
            MUAC_cat_enc = 0
        elif data.muac_cm < 12.5:
            MUAC_cat_enc = 1
        else:
            MUAC_cat_enc = 2

        # ---- build full feature dataframe ----
        X_new = pd.DataFrame([{
            "age_months": data.age_months,
            "height_cm": data.height_cm,
            "weight_kg": data.weight_kg,
            "muac_cm": data.muac_cm,
            "bmi": bmi,
            "sex_enc": sex_enc,
            "MUAC_cat_enc": MUAC_cat_enc
        }])

        # ---- FORCE training feature order (CRITICAL FIX) ----
        FEATURE_ORDER = list(imputer.feature_names_in_)
        X_new = X_new[FEATURE_ORDER]

        # ---- preprocessing ----
        X_new_imp = pd.DataFrame(
            imputer.transform(X_new),
            columns=FEATURE_ORDER
        )

        # ---- prediction ----
        pred_enc = model.predict(X_new_imp)
        prediction = label_encoder.inverse_transform(pred_enc)[0]

        # ---- AI suggestions (optional) ----
        # ai_text = generate_nutrition_suggestions(prediction)
        ai_text = generate_nutrition_suggestions(prediction)


        return {
            "prediction": prediction,
            "ai_suggestions": ai_text
        }

    except Exception as e:
        return {"error": str(e)}

