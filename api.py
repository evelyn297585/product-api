
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o poné solo la URL de Lovable si querés restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "API para clasificar productos está activa 🚀"}

@app.post("/predict")
def predict(product: dict):
    model = joblib.load("product_model.pkl")

    combined_text = (
        product.get("product_Name", "") + " " +
        product.get("Brand", "") + " " +
        product.get("Color", "") + " " +
        product.get("Variant_URL", "")
    )

    prediction = model.predict([combined_text])[0]

    return {"Predicted_Label": prediction}

