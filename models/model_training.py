import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    accuracy_score,
    classification_report
)
import wandb


# Initialize wandb
def init_wandb(entity_name="sam-vburgh-university-of-denver",project_name="toxic-comment-moderation-app", experiment_name=None):
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
        entity=entity_name,
        project=project_name,
        name=experiment_name,
        config={
            "dataset": "jigsaw-toxic-comment-classification-challenge",
            "model_type": "logistic-regression",
            "test_size": 0.2,
            "random_state": 42,
            "alpha": 1.0,
            "max_iter": 1000,
            "n_estimators": 100
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

def load_and_prepare_data():
    """
    
    """
    # Load dataset
    # read csv
    train_df = pd.read_csv("https://toxic-comment-moderation-app.s3.us-east-1.amazonaws.com/train.csv")
    test_df = pd.read_csv("https://toxic-comment-moderation-app.s3.us-east-1.amazonaws.com/test.csv")

    label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    X_train = train_df['comment_text']
    y_train = train_df[label_cols]

    X_test = test_df['comment_text']
    y_test = test_df[label_cols]

    
    
    # Log data split info
    wandb.log({
        "train_size": len(X_train),
        "test_size": len(X_test)
    })

    return X_train, X_test, y_train, y_test, label_cols
    
def build_lr_pipeline():
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1,2), 
            min_df=3,         
            strip_accents="unicode",
            sublinear_tf=True 
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced"
            ), n_jobs=-1))
    ])

    return lr_pipeline

def build_svm_pipeline():
    svm_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1,2),
            min_df=3,
            strip_accents="unicode",
            sublinear_tf=True
        )),
        ("clf", OneVsRestClassifier(
            LinearSVC(  # strong margins; fast on sparse text
                C=1.0,          # tune if you like: [0.25, 0.5, 1, 2]
                class_weight="balanced"  # helps rare labels
            ),
            n_jobs=-1
        ))
    ])

    return svm_pipeline

def train_and_log_model(pipeline, X_train, y_train, X_test, y_test, model_name, target_names):
    """
    Train a model and log metrics to W&B
    """
    print(f"Training {model_name}...")
    

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))

    # --- Micro / Macro ---
    micro_f1   = f1_score(y_test, y_pred, average="micro")
    macro_f1   = f1_score(y_test, y_pred, average="macro")

    micro_prec = precision_score(y_test, y_pred, average="micro")
    micro_rec  = recall_score(y_test, y_pred, average="micro")

    macro_prec = precision_score(y_test, y_pred, average="macro")
    macro_rec  = recall_score(y_test, y_pred, average="macro")
    
    # Calculate and log metrics
    wandb.log({
        "model": model_name,
        "micro/f1": micro_f1,
        "macro/f1": macro_f1,
        "micro/precision": micro_prec,
        "micro/recall": micro_rec,
        "macro/precision": macro_prec,
        "macro/recall": macro_rec
    })
    
    print(f" Training completed for {model_name}")

def main():
    """
    Main training pipeline - trains one model at a time
    """
    # Define models to train
    models = {
        "LogisticRegression": build_lr_pipeline,
        "SVM": build_svm_pipeline,
    }
    
    # Train each model in a separate W&B run
    for model_name, pipeline_function in models.items():
        # Initialize W&B for this specific model
        config = init_wandb(project_name="diabetes-prediction", 
                           experiment_name=f"{model_name}-experiment")
        
        # Update config with current model
        wandb.config.update({"model_name": model_name})
        
        # Load and prepare data
        print(f"\nStarting experiment for {model_name}")
        print("Loading and preparing data...")
        X_train, X_test, y_train, y_test, target_names = load_and_prepare_data(
            test_size=config.test_size,
            random_state=config.random_state
        )
        
        # Create model instance
        pipeline = pipeline_function()
        
        # Train and log
        train_and_log_model(pipeline, X_train, y_train, X_test, y_test, model_name, target_names)
        
        # Finish the W&B run
        wandb.finish()
        print(f"Experiment for {model_name} completed!\n")
    
    print("All experiments completed! Check your W&B dashboard for results.")

if __name__ == "__main__":
    main()
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
