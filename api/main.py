import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import json
from pathlib import Path
from datetime import datetime

# create app
app = FastAPI(
    title="Toxic Comment Moderation App",
)

# load model
model = None

def _load_model():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "sentiment_model.pkl",
        here.parent / "sentiment_model.pkl",
    ]
    for p in candidates:
        if p.is_file():
            return joblib.load(p)
    raise FileNotFoundError("Model not found")

try:
    model = _load_model()
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None


# create prediction request model
class PredictionRequest(BaseModel):
    text: str
    true_labels: str

# generate startup event
@app.on_event("startup")
def startup_event():
    """
    Startup event to print if model is loaded or not
    """
    if model is None:
        print("Warning: Model not loaded")
    else:
        print("Model loaded successfully")

# get health check endpoint
@app.get("/health")
def health_check():
    """
    Health check endpoint to verify if the API is running
    """
    if model is None:
        return {"status": "unhealthy", "message": "Model not loaded but app is running"}
    return {"status": "healthy", "message": "Model loaded successfully and app is running"}

# create predict endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Predict endpoint to predict the sentiment of the provided review text.
    Returns a JSON object with the predicted sentiment, "positive" or "negative"
    """
    # check if model is loaded  
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    
    # predict sentiment
    try:
        prediction = model.predict([request.text])
        
        # create log entry
        log = {
            "timestamp": datetime.now().isoformat(),
            "request_text": request.text,
            "predicted_sentiment": prediction[0],
            "true_sentiment": request.true_label
        }

        # write log entry
        with open("/logs/prediction_logs.json", "a") as f:
            f.write(json.dumps(log) + "\n")

        # return prediction
        return {"sentiment": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error predicting sentiment: {e}")
