from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="template")

print("Loading model...")
model = joblib.load("model/model.pkl")
print("Model loaded!")

class NewsInput(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(news: NewsInput):
    text = news.text.strip()
    if not text:
        return {"error": "No text provided"}

    prediction = model.predict([text])[0]
    decision = model.decision_function([text])[0]
    confidence = float(1 / (1 + np.exp(-abs(decision)))) * 100

    label = "REAL" if prediction == 1 else "FAKE"

    return {
        "label": label,
        "confidence": round(confidence, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)