import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path

def test_app_runs():
    app_path = Path(__file__).parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run(timeout=30)
    assert not at.exception

def test_submit_button_exists():
    app_path = Path(__file__).parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run(timeout=30)
    submit_buttons = [el for el in at.button if getattr(el, 'label', None) == 'Submit']
    assert submit_buttons, "Submit button not found in the app."