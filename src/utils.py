import logging
import sys


def get_logger(logger_name: str, path_to_log:str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    
    #config file for all of the logging messaages
    fh = logging.FileHandler(path_to_log, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_format = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    stream_handler.setFormatter(stream_format)
    logger.addHandler(stream_handler)
    return logger
