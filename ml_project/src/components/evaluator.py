import joblib
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def predict(filepath):
    logger.info("Loading trained model...")
    model = joblib.load("artifacts/model.joblib")

    df = pd.read_csv(filepath).drop(columns="charges", errors="ignore").head(5)
    preds = model.predict(df)

    logger.info(f"Predictions: {preds.tolist()}")
    return preds