from enum import Enum

from .wandb import WanDBWriter


def get_visualizer(config, logger, visualize):
    if visualize == "wandb":
        return WanDBWriter(config, logger)

    return None
