import wandb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib



warnings.filterwarnings('ignore')

# Initialize wandb
def init_wandb(project_name="toxic_comment_prediction", experiment_name=None):
    """
    Initialize a new W&B run
    """
    wandb.init(
        project=project_name,
        name=experiment_name,
        config={
            "dataset": "toxic_comments",
            # "model_type": "ridge",
            # "test_size": 0.2,
            # "random_state": 42,
            # "alpha": 1.0,
            # "max_iter": 1000,
            # "n_estimators": 100
        }
    )
    return wandb.config

def read_data():
    train_data = pd.read_csv('models/train.csv')
    x = train_data['comment_text'].fillna('')
    targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    y = train_data[targets]
    return x, y, targets

def train_model(x, y):
    vectorizer = CountVectorizer(stop_words='english')
    clf = MultiOutputClassifier(MultinomialNB(alpha=0.1))
    wandb.log({
        'vectorizer_type': vectorizer.__class__.__name__,
        'classifier_type': clf.__class__.__name__
    })
    pipe = Pipeline(steps=[
        ('vectorizer', vectorizer),
        ('clf', clf)
    ])

    return pipe.fit(x, y)

def dump_to_file(pipeline_object, targets, filename='models/toxic_comment_model.pkl'):
    model_data = {
        'model': pipeline_object,
        'target_columns': targets
    }
    joblib.dump(model_data, filename)
    print('model saved')

def make_predictions(model, test_file='models/test.csv'):
    """Make predictions on test set and save for submission"""
    test_data = pd.read_csv(test_file)
    test_comments = test_data['comment_text'].fillna('')
    
    predictions = model.predict_proba(test_comments)
    
    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    out_df = pd.DataFrame({'id': test_data['id']})
    
    for i, col in enumerate(target_cols):
        if predictions[i].shape[1] > 1: # predict proba occasional flips these? https://www.geeksforgeeks.org/machine-learning/understanding-the-predictproba-function-in-scikit-learns-svc/
            out_df[col] = predictions[i][:, 1]
        else:
            out_df[col] = predictions[i][:, 0]
    return out_df

if __name__ == '__main__':

    init_wandb()
    X, y, targets = read_data()
    
    print(f"Label distribution:")
    stratification = 'obscene'
    print(y.sum())
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y[stratification]
    )
    wandb.log({
        "train_size": len(X),
        'stratification': stratification,
        "toxic_pct": y['toxic'].mean(),
        "severe_toxic_pct": y['severe_toxic'].mean(),
        "obscene_pct": y['obscene'].mean(),
        "threat_pct": y['threat'].mean(),
        "insult_pct": y['insult'].mean(),
        "identity_hate_pct": y['identity_hate'].mean(),
    })
    model = train_model(x_train, y_train)
    dump_to_file(model, targets)
    predictions_log = make_predictions(model)
    test_labels = pd.read_csv('models/test_labels.csv').replace(-1, 1)
    for target in targets:
        wandb.log({
            f"{target}_accuracy": accuracy_score(test_labels[target], predictions_log[target] > 0.5),
            f"{target}_f1": f1_score(test_labels[target], predictions_log[target] > 0.5),
            f"{target}_precision": precision_score(test_labels[target], predictions_log[target] > 0.5),
            f"{target}_recall": recall_score(test_labels[target], predictions_log[target] > 0.5),
        })

    model_artifact = wandb.Artifact(
        'comments_model',
        type='model',
        description='Toxic comment classification model'
    )
    model_artifact.add_file('models/toxic_comment_model.pkl')
    wandb.log_artifact(model_artifact)
    wandb.finish()