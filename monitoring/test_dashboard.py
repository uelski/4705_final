from streamlit.testing.v1 import AppTest
from pathlib import Path
from unittest.mock import patch, MagicMock

@patch("boto3.resource")
def test_monitoring_app_runs(mock_boto_resource):
    # fetch_all_logs() mocking
    mock_table = MagicMock()
    mock_table.scan.return_value = {
        "Items": [
            {
                "timestamp": "2025-08-21T10:00:00Z",
                "request_text": "Example input",
                "response": {
                    "toxic": 1,
                    "severe_toxic": 0,
                    "obscene": 1,
                    "threat": 0,
                    "insult": 1,
                    "identity_hate": 0
                },
                "true_labels": {
                    "toxic": 1,
                    "severe_toxic": 0,
                    "obscene": 1,
                    "threat": 0,
                    "insult": 1,
                    "identity_hate": 0
                }
            }
        ]
    }

    mock_ddb = MagicMock()
    mock_ddb.Table.return_value = mock_table
    mock_boto_resource.return_value = mock_ddb

    # base streamlit test.
    app_path = Path(__file__).parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run(timeout=30)

    assert not at.exception