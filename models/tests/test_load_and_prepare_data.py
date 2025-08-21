import pytest
import pandas as pd

from model_training import load_and_prepare_data

LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

@pytest.fixture
def fake_dataframes():
    # Minimal but realistic fixtures
    train_df = pd.DataFrame({
        "comment_text": ["you rock", "you stink", "meh"],
        "toxic": [0, 1, 0],
        "severe_toxic": [0, 0, 0],
        "obscene": [0, 1, 0],
        "threat": [0, 0, 0],
        "insult": [0, 1, 0],
        "identity_hate": [0, 0, 0],
    })

    test_df = pd.DataFrame({
        "comment_text": ["ok...", "the best", "awful take"],
    })

    # Include -1 values to verify replacement to 1
    test_labels_df = pd.DataFrame({
        "toxic": [-1, 0, 1],
        "severe_toxic": [0, -1, 0],
        "obscene": [1, 0, -1],
        "threat": [0, 0, 0],
        "insult": [-1, -1, 0],
        "identity_hate": [0, 0, -1],
    })

    return train_df, test_df, test_labels_df

@pytest.fixture
def patch_read_csv(monkeypatch, fake_dataframes):
    train_df, test_df, test_labels_df = fake_dataframes

    def fake_read_csv(path, *args, **kwargs):
        s = str(path)
        if "train.csv" in s: 
            return train_df.copy()
        if "test.csv" in s: 
            return test_df.copy()
        if "test_labels.csv" in s: 
            return test_labels_df.copy()
        raise AssertionError(f"Unexpected path {path}")

    import model_training
    monkeypatch.setattr(model_training.pd, "read_csv", fake_read_csv)

def test_load_and_prepare_data_happy_path(patch_read_csv):
    X_train, X_test, y_train, y_test, label_cols = load_and_prepare_data()

    # Labels come back as expected and in the right order
    assert label_cols == LABELS
    assert len(X_train) == len(y_train) == 3
    assert len(X_test) == len(y_test) == 3
    assert list(y_train.columns) == LABELS
    assert list(y_test.columns) == LABELS
    assert not (y_test.values == -1).any()
    assert y_test.iloc[0]["toxic"] == 1