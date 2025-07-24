from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import time
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Removes or clips outliers in numeric columns using the IQR method.
    Clips values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_ = pd.DataFrame(X).copy()
        for col in X_.select_dtypes(include='number').columns:
            q1 = X_[col].quantile(0.25)
            q3 = X_[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.multiplier * iqr
            upper = q3 + self.multiplier * iqr
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_ = pd.DataFrame(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            X_[col] = X_[col].clip(lower, upper)
        return X_

def train(X_train, y_train, X_test, y_test):
    
    mlflow.set_experiment("Medical_Insurance_Charges_Regression")

    # Identify feature types
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    logger.info(f"Numerical columns: {num_cols}")
    logger.info(f"Categorical columns: {cat_cols}")

    # Preprocessor: outlier removal + scale numeric + encode categorical
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("outlier_remover", OutlierRemover()),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    # Base model
    rf = RandomForestRegressor(random_state=42)

    # Pipeline: preprocess + model
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", rf)
    ])

    # Hyperparameter tuning grid
    param_grid = {
        "regressor__n_estimators": [100, 200, 500],
        "regressor__max_depth": [None, 10, 20, 30],
        "regressor__min_samples_split": [2, 5, 10],
        "regressor__min_samples_leaf": [1, 2, 4],
        "regressor__max_features": ["sqrt", "log2"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    with mlflow.start_run():
        logger.info("Starting training for regression with outlier handling and hyperparameter tuning...")

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        elapsed_time = time.time() - start_time

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        y_pred = best_pipeline.predict(X_test)

        # Regression metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f"Training completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"R² Score: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "r2_score": r2,
            "mse": mse,
            "mae": mae,
            "training_time_seconds": elapsed_time
        })
        mlflow.sklearn.log_model(best_pipeline, "model")

        # Save locally
        joblib.dump(best_pipeline, "artifacts/model.joblib")
        logger.info("Best pipeline saved to artifacts/model.joblib")

    logger.info("Regression training pipeline finished.")
