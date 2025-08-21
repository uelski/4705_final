import pytest
import random
import numpy as np
import pandas as pd

LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

@pytest.fixture(autouse=True)
def set_seeds():
    random.seed(1337)
    np.random.seed(1337)

@pytest.fixture
def label_cols():
    return LABELS

@pytest.fixture
def dummy_text_train():
    return pd.Series([
        "this movie was great and fun",
        "awful insult and very toxic",
        "threat implied but unclear",
        "clean friendly and kind",
        "obscene words here",
        "identity hate speech sample",
        "neutral comment with nothing bad",
        "bad acting but great plot",
    ])

@pytest.fixture
def dummy_text_test():
    return pd.Series([
        "utterly great performance",
        "what a terrible insult",
        "neutral vibes only",
    ])

@pytest.fixture
def dummy_labels_train():
    return pd.DataFrame({
        "toxic":         [0,1,0,0,1,1,0,0],
        "severe_toxic":  [0,0,0,0,0,1,0,0],
        "obscene":       [0,1,0,0,1,0,0,0],
        "threat":        [0,0,1,0,0,0,0,0],
        "insult":        [0,1,0,0,0,1,0,0],
        "identity_hate": [0,0,0,0,0,1,0,0],
    })


@pytest.fixture
def fitted_nb_pipeline(dummy_text_train, dummy_labels_train):
    from model_training import build_nb_pipeline
    pipe = build_nb_pipeline()
    pipe.fit(dummy_text_train, dummy_labels_train)
    return pipe

@pytest.fixture
def fitted_lr_pipeline(dummy_text_train, dummy_labels_train):
    from model_training import build_lr_pipeline
    pipe = build_lr_pipeline() 
    pipe.fit(dummy_text_train, dummy_labels_train)
    return pipe

@pytest.fixture
def fitted_svm_pipeline(dummy_text_train, dummy_labels_train):
    from model_training import build_svm_pipeline
    pipe = build_svm_pipeline() 
    pipe.fit(dummy_text_train, dummy_labels_train)
    return pipe
