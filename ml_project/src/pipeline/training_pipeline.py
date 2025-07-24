from src.components.data_loader import load_data
from src.components.trainer import train


from src.utils.logger import logger

def run_training():
    logger.info("Training pipeline started")
    # … your training code …
    logger.info("Training pipeline completed")

if __name__ == "__main__":
    run_training()

def main():
    X_train, X_test, y_train, y_test = load_data("data/insurance.csv")
    train(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()