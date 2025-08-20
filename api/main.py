import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import json
from pathlib import Path
from datetime import datetime
import wandb
import os

# create app
app = FastAPI(
    title="Toxic Comment Moderation App",
)

model = None

def _load_model():
    api_key = json.load(open("secrets.json"))["wandb_api_key"]
    wandb.login(key=api_key)
    run = wandb.init(project="toxic_comment_prediction")
    artifact = run.use_artifact(artifact_or_name="log_reg_model:latest") # this creates a reference within Weights & Biases that this artifact was used by this run.
    path = artifact.download() # this downloads the artifact from Weights & Biases to your local system where the code is executing.

    for filename in os.listdir(path):
        if filename.endswith(".joblib"):
            model_path = os.path.join(path, filename)
            break
    else:
        raise FileNotFoundError("No .joblib file found in artifact directory.")

    run.finish()
    model = joblib.load(model_path)
    print(type(model))
    return model

try:
    model = _load_model()
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

# create prediction request model
class PredictionRequest(BaseModel):
    text: str

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
        prediction = model.predict([request.text])[0]
        prediction_output = {
            f"sentiment_{i}": int(val) for i, val in enumerate(prediction)
        }
        print('prediction output: ', prediction_output)
        # create log entry
        log = {
            "timestamp": datetime.now().isoformat(),
            "request_text": request.text,
            'response': prediction_output
        }

        # write log entry
        # with open("/logs/prediction_logs.json", "a") as f:
        #     f.write(json.dumps(log) + "\n")
        return prediction_output
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error predicting sentiment: {e}")
