from streamlit.testing.v1 import AppTest
from pathlib import Path

def test_app_runs():
    app_path = Path(__file__).parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run(timeout=30)
    assert not at.exception