# -*- coding: utf-8 -*-

from logging import DEBUG, Formatter, Logger, StreamHandler, getLogger


def initialize_logging(name: str = __name__) -> Logger:
    logger = getLogger(name)
    logger.propagate = False
    logger.setLevel(DEBUG)

    handler = StreamHandler()
    handler.setFormatter(
        Formatter(
            "[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] \n %(message)s"
        )
    )
    handler.setLevel(DEBUG)
    logger.addHandler(handler)

    return logger
