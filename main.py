# -*- coding: utf-8 -*-

"""Main file to launch the experiments based on configuration.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import warnings

import hydra
from omegaconf import DictConfig

from configs import config_names
from core.environments import generate_environment, generate_test_environment
from core.handlers.evaluators import generate_evaluator
from core.handlers.trainers import generate_trainer
from core.maps import generate_map
from core.utils.logging import initialize_logging
from core.utils.seed import set_seed
from core.worlds import generate_world

warnings.simplefilter("ignore")
logger = initialize_logging(__name__)


@hydra.main(config_path="configs", config_name=config_names["mat"])
def main(config: DictConfig):
    set_seed(seed=config.seed)
    world_map = generate_map(config=config)
    world = generate_world(config=config, world_map=world_map)

    if config.phase == "training":
        env = generate_environment(config=config, world=world)
        trainer = generate_trainer(config=config, environment=env)
        trainer.run()

    # TODO implement evaluation handler for analysis and test.
    elif config.phase == "evaluation":
        env = generate_test_environment(config=config, world=world)
        evaluator = generate_evaluator(config=config, environment=env)
        evaluator.run()

    else:
        logger.warn(f"Unexpected phase is given. config.phase: {config.phase}")

        raise ValueError()


if __name__ == "__main__":
    main()
