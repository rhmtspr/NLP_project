from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

class EmailRequest(BaseModel):
    message: str

@app.post("/predict")
async def predict(data: EmailRequest):
    vector = vectorizer.transform([data.message])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    spam_prob = round(proba[1] * 100, 2)
    ham_prob = round(proba[0] * 100, 2)
    return {
        "prediction": prediction,
        "spam_prob": spam_prob,
        "ham_prob": ham_prob,
        }