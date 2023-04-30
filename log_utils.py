import logging
import os
import sys 

LOG_DIR = "log"

def create_logger(name: str, dir: str) -> logging.Logger:
    """Create a logger that log on file and console"""
    if logger_exists(name):
        return logging.getLogger(name)

    # Create the logger 
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create the handler for the file 
    file_handler = logging.FileHandler(os.path.join(dir, name + "-log.txt"))
    file_handler.setLevel(logging.DEBUG)
     
    # Create the handler for the console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create the formatter to be used by file and console
    formatter = logging.Formatter("[%(asctime)s]: %(message)s")

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Register the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    if logger_exists(name):
        return logging.getLogger(name)
    else:
        return None


def logger_exists(name: str) -> bool:
    if name in logging.Logger.manager.loggerDict.keys():
        return True
    return False