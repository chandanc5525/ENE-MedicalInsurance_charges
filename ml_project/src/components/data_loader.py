import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_data(filepath):
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    X = df.drop(columns="charges")
    y = df["charges"]
    return train_test_split(X, y, test_size=0.3, random_state=42)