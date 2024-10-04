# -*- coding: utf-8 -*-

from logging import INFO, Formatter, Logger, StreamHandler, getLogger


def initialize_logging(name: str = __name__) -> Logger:
    logger = getLogger(name)
    logger.propagate = False
    logger.setLevel(INFO)

    handler = StreamHandler()
    handler.setFormatter(
        Formatter(
            "[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] \n %(message)s"
        )
    )
    handler.setLevel(INFO)
    logger.addHandler(handler)

    return logger
