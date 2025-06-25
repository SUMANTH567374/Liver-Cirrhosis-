# src/models/train_models.py

import os
import yaml
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Logger setup
def setup_logger(log_file):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# Initialize paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_path = os.path.join(BASE_DIR, "data", "final", "balanced_data.csv")
params_path = os.path.join(BASE_DIR, "params.yaml")
models_dir = os.path.join(BASE_DIR, "models")
logs_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
logger = setup_logger(os.path.join(logs_dir, "train_models.log"))

try:
    # Load params.yaml
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    # Load data
    balanced_data = pd.read_csv(data_path)
    logger.info(f"Loaded data with shape {balanced_data.shape}")

    X = balanced_data.drop(columns=['Status'])
    y = balanced_data['Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['split']['test_size'],
        stratify=y,
        random_state=params['split']['random_state']
    )

    # Model config from params.yaml
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=params['logistic_regression']['max_iter']),
            {
                "clf__C": params['logistic_regression']['C'],
                "clf__penalty": params['logistic_regression']['penalty'],
                "clf__solver": params['logistic_regression']['solver']
            }
        ),
        "RandomForest": (
            RandomForestClassifier(),
            {
                "clf__n_estimators": params['random_forest']['n_estimators'],
                "clf__max_depth": params['random_forest']['max_depth'],
                "clf__min_samples_split": params['random_forest']['min_samples_split'],
                "clf__min_samples_leaf": params['random_forest']['min_samples_leaf']
            }
        ),
        "SVM": (
            SVC(probability=True),
            {
                "clf__C": params['svm']['C'],
                "clf__kernel": params['svm']['kernel'],
                "clf__gamma": params['svm']['gamma']
            }
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": params['knn']['n_neighbors'],
                "clf__weights": params['knn']['weights'],
                "clf__metric": params['knn']['metric']
            }
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(),
            {
                "clf__n_estimators": params['gradient_boosting']['n_estimators'],
                "clf__learning_rate": params['gradient_boosting']['learning_rate'],
                "clf__max_depth": params['gradient_boosting']['max_depth'],
                "clf__subsample": params['gradient_boosting']['subsample']
            }
        )
    }

    for model_name, (model, grid_params) in models.items():
        logger.info(f"Training {model_name}...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model)
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid=grid_params,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        model_path = os.path.join(models_dir, f"{model_name}_best.pkl")
        joblib.dump(grid.best_estimator_, model_path)
        logger.info(f"Saved {model_name} model to {model_path}")

except Exception as e:
    logger.error(f"Training pipeline failed: {str(e)}")
