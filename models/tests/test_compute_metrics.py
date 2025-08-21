import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, accuracy_score

from model_training import compute_and_log_metrics

LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

class FakeRun:
    def __init__(self):
        self.logged = None
        self.summary = {}
    def log(self, d):
        self.logged = d.copy()

def test_compute_and_log_metrics_happy_path():
    y_true = pd.DataFrame([[0,0,0,0,1,1],[1,1,1,1,1,1],[1,0,1,0,1,0],[0,1,0,1,0,1]], columns=LABELS)
    y_pred = pd.DataFrame([[0,0,0,0,1,1],[1,1,1,1,1,0],[1,1,1,0,1,0],[0,1,0,1,0,1]], columns=LABELS) 
    run = FakeRun()

    out = compute_and_log_metrics(y_true, y_pred, run, LABELS)

    for k in ["micro/f1","macro/f1","micro/precision","micro/recall","macro/precision","macro/recall","subset_accuracy"]:
        assert k in out
    for lbl in LABELS:
        assert f"f1/{lbl}" in out

    assert np.isclose(out["micro/f1"], f1_score(y_true, y_pred, average="micro", zero_division=0))
    assert np.isclose(out["macro/precision"], precision_score(y_true, y_pred, average="macro", zero_division=0))
    assert np.isclose(out["subset_accuracy"], accuracy_score(y_true, y_pred))

    per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    for lbl, val in zip(LABELS, per_label):
        assert np.isclose(out[f"f1/{lbl}"], val)

    assert run.logged is not None and run.logged == out
    assert run.summary == out

