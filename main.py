"""Main file to launch the experiments based on configuration.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import os
import warnings

import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from omegaconf import DictConfig

from configs import config_names
from core.environments import generate_environment
from core.handlers.evaluators import generate_evaluator
from core.handlers.trainers import generate_trainer
from core.maps import generate_map
from core.utils.logging import initialize_logging
from core.utils.seed import set_seed
from core.worlds import generate_world

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
sns.set()

warnings.simplefilter("ignore")

logger = initialize_logging(__name__)

@hydra.main(config_path="configs", config_name=config_names["PosNeg"])
def main(config: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    set_seed(seed=config.seed)
    world_map = generate_map(config=config)
    world = generate_world(config=config, world_map=world_map)
    env = generate_environment(config=config, world=world)

    if config.phase == "training":
        trainer = generate_trainer(config=config, environment=env)

        try:
            trainer.run()

        finally:
            for agent_id in range(len(trainer.agents)):
                for ch_id, heatmap in enumerate(trainer.env.observation_stats[agent_id]):
                    if ch_id == 7:
                        heatmap = heatmap * -1
                    fig = plt.figure()
                    sns.heatmap(
                        heatmap.T,
                        square=True,
                        # annot=True,
                        # annot_kws={"fontsize": 6},
                    )

                    wandb.log(
                        {
                            f"agent_{str(agent_id)}/observation_stats_final_{str(ch_id)}": [
                                wandb.Image(
                                    data_or_path=fig,
                                    caption=f"observation stats ({str(ch_id)})",
                                )
                            ]
                        }
                    )
                    logger.info(f"final/aegnt_id({agent_id})/ch_id({ch_id}): {heatmap}")

            trainer.save_state_dict()

    elif config.phase == "evaluation":
        evaluator = generate_evaluator(config=config, environment=env)
        evaluator.run()

    else:
        logger.warn(f"Unexpected phase is given. config.phase: {config.phase}")

        raise ValueError()


if __name__ == "__main__":
    main()
