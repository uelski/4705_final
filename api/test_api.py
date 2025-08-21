import os
import pytest
import main
from fastapi.testclient import TestClient
import warnings
from unittest.mock import patch, MagicMock

warnings.filterwarnings("ignore")
client = TestClient(main.app)

def mock_predict(input):
    # Return a list with a single prediction, e.g., all zeros
    return [[0, 0, 0, 0, 0, 0]]

@patch.object(main, 'model', MagicMock(predict=mock_predict))
def test_response_structure():
    main.startup_event()
    request_text = 'nice love happy good'
    response = client.post("/predict", json={"text": request_text})
    assert response.status_code == 200
    response = response.json()
    assert isinstance(response['timestamp'], str)
    assert isinstance(response['response'], dict)
    assert (isinstance(response['true_labels'], dict) or response['true_labels'] is None)
    assert response['request_text'] == request_text

@patch.object(main, 'model', MagicMock(predict=mock_predict))
def test_predict_positive():
    main.startup_event()
    request_text = 'nice love happy good'
    response = client.post("/predict", json={"text": request_text})
    response = response.json()
    assert response['response']['toxic'] == 0
    assert response['response']['severe_toxic'] == 0
    assert response['response']['obscene'] == 0
    assert response['response']['threat'] == 0
    assert response['response']['insult'] == 0
    assert response['response']['identity_hate'] == 0

def mock_predict(input):
    # Return a list with a single prediction, e.g., all zeros
    return [[1, 0, 0, 0, 0, 0]]

@patch.object(main, 'model', MagicMock(predict=mock_predict))
def test_predict_negative():
    main.startup_event()
    request_text = 'terrible stupid ass bad disgusting shit fuck evil nasty crap'
    response = client.post("/predict", json={"text": request_text})
    response = response.json()
    assert (response['response']['toxic'] == 1 or
            response['response']['severe_toxic'] == 1 or
            response['response']['obscene'] == 1 or
            response['response']['threat'] == 1 or
            response['response']['insult'] == 1 or
            response['response']['identity_hate'] == 1)

@patch.object(main, 'model', MagicMock(predict=mock_predict))
def test_health():
    main.startup_event()
    response = client.get("/health")
    assert response.json() == {
                "status": "healthy",
                "message": "Model loaded successfully and app is running"
            }
    assert response.status_code == 200