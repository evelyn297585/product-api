from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProductInput(BaseModel):
    product_Name: str
    Brand: str
    Color: str
    Variant_URL: str

@app.get("/")
def home():
    return {"message": "API para clasificar productos está activa 🚀"}

@app.post("/predict")
def predict(product: ProductInput):
    model = joblib.load("product_model.pkl")
    combined_text = (
        product.product_Name + " " +
        product.Brand + " " +
        product.Color + " " +
        product.Variant_URL
    )
    prediction = model.predict([combined_text])[0]
    return {"Predicted_Label": prediction}

