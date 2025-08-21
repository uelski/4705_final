import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import json
from datetime import datetime
import wandb
import os
import boto3

# create app
app = FastAPI(
    title="Toxic Comment Moderation App",
)

# Environment variables for DynamoDB
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "table_01")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

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
    true_labels: dict[str, int] = None  # expects keys: "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"

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
        labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        prediction_output = {}
        for i, label in enumerate(labels):
            prediction_output[label] = int(prediction[i])
        print('prediction output: ', prediction_output)
        # create log entry
        log = {
            "timestamp": datetime.now().isoformat(),
            "request_text": request.text,
            'response': prediction_output,
            'true_labels': request.true_labels
        }
        # write log entry to DynamoDB
        try:
            dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
            table = dynamodb.Table(DYNAMODB_TABLE_NAME)
            table.put_item(
                Item = log
            )
        except Exception as db_error:
            print(f'Error saving prediction to DynamoDB: {db_error}')
        return log

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error predicting sentiment: {e}")
