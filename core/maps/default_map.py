
from omegaconf import DictConfig

from .abstract_map import AbstractMap


class DefaultMap(AbstractMap):
    def __init__(self, config: DictConfig):
        super().__init__(config=config)
