import wandb
import pandas as pd
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import uuid

warnings.filterwarnings('ignore')

# Initialize wandb
def init_wandb(project_name="toxic_comment_prediction", experiment_name=None, config_name="", config_registry="", group=None):
    """
    Initialize a new W&B run
    """
    run = wandb.init(
        project=project_name,
        name=experiment_name,
        group=group,
        reinit=True,
        config={
            "dataset": "toxic_comments",
            "model_name": config_name,
            "registry_name": config_registry
        }
    )
    return run

def load_and_prepare_data():
    """
    load the training data, test data and test labels from aws s3 bucket
    """
    # read csv
    print('load and prepare data')
    train_df = pd.read_csv("https://toxic-comment-moderation-app.s3.us-east-1.amazonaws.com/train.csv")
    test_df = pd.read_csv("https://toxic-comment-moderation-app.s3.us-east-1.amazonaws.com/test.csv")
    test_labels_df = pd.read_csv("https://toxic-comment-moderation-app.s3.us-east-1.amazonaws.com/test_labels.csv").replace(-1, 1)

    label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    X_train = train_df['comment_text']
    y_train = train_df[label_cols]

    X_test = test_df['comment_text']
    y_test = test_labels_df[label_cols]

    return X_train, X_test, y_train, y_test, label_cols


def build_nb_pipeline():
    """
    build Multinomial NaiveBayes model pipeline
    """
    print('building nb pipeline')
    vectorizer = CountVectorizer(stop_words='english')
    clf = MultiOutputClassifier(MultinomialNB(alpha=0.1))
    nb_pipeline = Pipeline(steps=[
        ('vectorizer', vectorizer),
        ('clf', clf)
    ])

    return nb_pipeline

def build_lr_pipeline():
    """
    build LogisticRegression model pipeline
    """
    print('building lr pipeline')
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
    """
    build the LinearSVC model pipeline
    """
    print('building svm pipeline')
    svm_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1,2),
            min_df=3,
            strip_accents="unicode",
            sublinear_tf=True
        )),
        ("clf", OneVsRestClassifier(
            LinearSVC(  
                C=1.0,   
                class_weight="balanced"
            ),
            n_jobs=-1
        ))
    ])

    return svm_pipeline

def train_model(pipeline, X_train, y_train, X_test, model_name):
    """
    Train a model 
    """
    print(f"Training {model_name}...")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    return y_pred

def compute_and_log_metrics(y_true, y_pred, run, label_cols):
    """
    calculate and send metrics to wandb.ai dashboard
    """
    print('computing metrics')
    metrics = {
        "micro/f1":       f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro/f1":       f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro/precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro/recall":   recall_score(y_true, y_pred, average="micro", zero_division=0),
        "macro/precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro/recall":   recall_score(y_true, y_pred, average="macro", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred), 
    }

    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics_to_log = {**metrics}
    for lbl, f1v in zip(label_cols, per_label_f1):
        metrics_to_log[f"f1/{lbl}"] = float(f1v)
    
    run.log(metrics_to_log)
    run.summary.update(metrics_to_log)

    return metrics_to_log

def build_model_artifact(model_name, pipeline, registry_name, label_cols, metrics, run):
    """
    build model, dump locally and use local path to create W&B artifact and log
    """
    print(f'build artifact for {model_name}')

    model_path = f"models/{model_name}.joblib"
    joblib.dump(pipeline, model_path)

    # Create + log artifact (versioned)
    art = wandb.Artifact(
        name=registry_name,
        type="model",
        metadata={
            "model_key": model_name,
            "labels": label_cols,
            "metrics": metrics
        }
    )
    art.add_file(model_path)

    # mark this version as a "candidate" (and as "latest" for convenience)
    run.log_artifact(art, aliases=["candidate", "latest", model_name])

def main():
    """
    Main training pipeline - trains one model at a time
    """
    # Define models to train
    models = {
        "log_reg": {
            "pipeline": build_lr_pipeline,
            "registry_name": "log_reg_model"
        },
        "linear_svm": {
            "pipeline": build_svm_pipeline,
            "registry_name": "linear_svm_model"
        },
        "multi_nb": {
            "pipeline": build_nb_pipeline,
            "registry_name": "multi_nb_model"
        }
    }

    # load data
    X_test, X_train, y_test, y_train, label_cols = load_and_prepare_data()

    # init results for viewing model performance together:
    results = []
    
    # Train each model in a separate W&B run
    for model_name, model_data in models.items():
        print(f'start of loop for {model_name}')
        # setup variables from model data
        pipeline = model_data["pipeline"]() 
        registry_name = model_data["registry_name"]

        # Initialize W&B for this specific model
        group_id = str(uuid.uuid4())
        run = init_wandb(project_name="toxic_comment_prediction", experiment_name=f"{model_name}-experiment", config_name=model_name, config_registry=registry_name, group=group_id)

        # log data metrics
        wandb.log({
            "train_size": len(X_train),
            "toxic_pct": y_train['toxic'].mean(),
            "severe_toxic_pct": y_train['severe_toxic'].mean(),
            "obscene_pct": y_train['obscene'].mean(),
            "threat_pct": y_train['threat'].mean(),
            "insult_pct": y_train['insult'].mean(),
            "identity_hate_pct": y_train['identity_hate'].mean(),
        })
        
        # Train model
        y_pred = train_model(pipeline, X_train, y_train, X_test, model_name)

        # compute and log metrics
        metrics = compute_and_log_metrics(y_test, y_pred, run, label_cols)

        # create model and artifact
        build_model_artifact(model_name, pipeline, registry_name, label_cols, metrics, run)
        
        # add to results to compare
        results.append({"key": model_name, "registry_name": registry_name, **metrics})
        # Finish the W&B run
        run.finish()
        print(f"Experiment for {model_name} completed!\n")

    print("All experiments completed! Check your W&B dashboard for results.")
    results_df = pd.DataFrame(results).sort_values("macro/f1", ascending=False)
    print(results_df[["key","micro/f1","macro/f1"]])

if __name__ == '__main__':
    main()

