import wandb
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
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
            "model_type": "ridge",
            "test_size": 0.2,
            "random_state": 42,
            "alpha": 1.0,
            "max_iter": 1000,
            "n_estimators": 100
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
    vectorizer = TfidfVectorizer(stop_words='english')
    clf = MultiOutputClassifier(MultinomialNB(alpha=0.1))

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

if __name__ == '__main__':
    x, y, targets = read_data()
    
    print(f"Label distribution:")
    print(y.sum())
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y['toxic']
    )

    model = train_model(x_train, y_train)
    dump_to_file(model, targets)





# def load_and_prepare_data(test_size=0.2, random_state=42):
#     """
#     Load diabetes dataset and prepare train/test splits
#     """
#     # Load dataset
#     X, y, targets = read_data()
    
#     # Log dataset info to W&B
#     wandb.log({
#         "train_size": len(X),
#         "toxic_pct": y['toxic'].mean(),
#         "severe_toxic_pct": y['severe_toxic'].mean(),
#         "obscene_pct": y['obscene'].mean(),
#         "threat_pct": y['threat'].mean(),
#         "insult_pct": y['insult'].mean(),
#         "identity_hate_pct": y['identity_hate'].mean(),
#     })

#     return X, y, targets

# def train_and_log_model(model, X_train, y_train, X_test, y_test, model_name, feature_names):
#     """
#     Train a model and log metrics to W&B
#     """
#     print(f"Training {model_name}...")
    
#     # Train the model
#     model.fit(X_train, y_train)
    
#     # Make predictions
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)
    
#     # Calculate and log metrics
#     wandb.log({
#         "model": model_name,
#         "train_mse": mean_squared_error(y_train, y_train_pred),
#         "test_mse": mean_squared_error(y_test, y_test_pred),
#         "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
#         "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
#         "train_mae": mean_absolute_error(y_train, y_train_pred),
#         "test_mae": mean_absolute_error(y_test, y_test_pred),
#         "train_r2": r2_score(y_train, y_train_pred),
#         "test_r2": r2_score(y_test, y_test_pred)
#     })
    
#     # Cross-validation metrics
#     cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
#                                 scoring='neg_mean_squared_error')
#     wandb.log({
#         "cv_rmse_mean": np.sqrt(-cv_scores.mean()),
#         "cv_rmse_std": np.sqrt(cv_scores.std())
#     })
    
#     # Residual statistics
#     residuals_train = y_train - y_train_pred
#     residuals_test = y_test - y_test_pred
    
#     wandb.log({
#         "train_residuals_mean": residuals_train.mean(),
#         "train_residuals_std": residuals_train.std(),
#         "test_residuals_mean": residuals_test.mean(),
#         "test_residuals_std": residuals_test.std()
#     })
    
#     # Prediction statistics
#     wandb.log({
#         "train_pred_mean": y_train_pred.mean(),
#         "train_pred_std": y_train_pred.std(),
#         "test_pred_mean": y_test_pred.mean(),
#         "test_pred_std": y_test_pred.std()
#     })
    
#     # Log feature importances for tree-based models
#     if hasattr(model, 'feature_importances_'):
#         for feature, importance in zip(feature_names, model.feature_importances_):
#             wandb.log({f"feature_importance_{feature}": importance})
    
#     # Log coefficients for linear models
#     if hasattr(model, 'coef_'):
#         for feature, coef in zip(feature_names, model.coef_):
#             wandb.log({f"coefficient_{feature}": coef})
        
#         if hasattr(model, 'intercept_'):
#             wandb.log({"intercept": model.intercept_})
    
#     print(f"  Training completed for {model_name}")

# def main():
#     """
#     Main training pipeline - trains one model at a time
#     """
#     x, y, targets = read_data()
#     # Define models to train
#     models = {
#         "Ridge": lambda config: Ridge(alpha=config.alpha, random_state=config.random_state),
#         "Lasso": lambda config: Lasso(alpha=config.alpha, random_state=config.random_state, max_iter=config.max_iter),
#         "ElasticNet": lambda config: ElasticNet(alpha=config.alpha, random_state=config.random_state, max_iter=config.max_iter),
#         "RandomForest": lambda config: RandomForestRegressor(n_estimators=config.n_estimators, random_state=config.random_state),
#         "GradientBoosting": lambda config: GradientBoostingRegressor(n_estimators=config.n_estimators, random_state=config.random_state)
#     }
    
#     # Train each model in a separate W&B run
#     for model_name, model_func in models.items():
#         # Initialize W&B for this specific model
#         config = init_wandb(project_name="diabetes-prediction", 
#                            experiment_name=f"{model_name}-experiment")
        
#         # Update config with current model
#         wandb.config.update({"model_name": model_name})
        
#         # Load and prepare data
#         print(f"\nStarting experiment for {model_name}")
#         print("Loading and preparing data...")
#         X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(
#             test_size=config.test_size,
#             random_state=config.random_state
#         )

#         X_train, X_test, y_train, y_test = train_test_split(
#             x, y, test_size=0.2, random_state=42, stratify=y['toxic']
#         )
#         # Create model instance
#         model = model_func(config)
        
#         # Train and log
#         train_and_log_model(model, X_train, y_train, X_test, y_test, model_name, feature_names)
        
#         # Finish the W&B run
#         wandb.finish()
#         print(f"Experiment for {model_name} completed!\n")
    
#     print("All experiments completed! Check your W&B dashboard for results.")

# if __name__ == "__main__":
#     main()