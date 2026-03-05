import logging
import datetime

def create_logger(name: str, prefix: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = logging.getLogger(f"{name}_{timestamp}")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(f"{prefix}_{timestamp}.log", mode="w")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger