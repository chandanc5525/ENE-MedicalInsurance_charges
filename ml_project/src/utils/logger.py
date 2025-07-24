import logging

def get_logger(name=__name__):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(name)

from loguru import logger
import os

os.makedirs("logs", exist_ok=True)
logger.add("logs/app.log", rotation="1 MB")