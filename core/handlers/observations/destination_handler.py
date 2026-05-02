"""Destination area handler for managing different destination matrix types.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig

from core.worlds import AbstractWorld
from core.utils.logging import initialize_logging

logger = initialize_logging(__name__)


class DestinationAreaHandler(ABC):
    """Abstract base class for destination area handlers."""
    
    def __init__(self, config: DictConfig, world: AbstractWorld):
        self.config = config
        self.world = world
    
    @abstractmethod
    def get_destination(self, agent_id: int) -> torch.Tensor:
        """Get destination tensor for a specific agent.
        
        Args:
            agent_id: The ID of the agent
            
        Returns:
            torch.Tensor: The destination area tensor for the agent
        """
        raise NotImplementedError()


class StandardDestinationHandler(DestinationAreaHandler):
    """Handler for standard 3D destination matrix (num_agents, SIZE_X, SIZE_Y)."""
    def get_destination(self, agent_id: int) -> torch.Tensor:
        """Get destination from standard matrix format."""
        return torch.from_numpy(
            deepcopy(self.world.map.destination_area_matrix[agent_id, :, :])
        )


class TensorDestinationHandler(DestinationAreaHandler):
    """Handler for tensor-based destination matrix (num_agents, tensor_features)."""
    def get_destination(self, agent_id: int) -> torch.Tensor:
        """Get destination from tensor-based matrix format."""
        return torch.from_numpy(
            deepcopy(self.world.map.destination_area_vector[agent_id, :])
        )


def generate_destination_handler(
    config: DictConfig, world: AbstractWorld
) -> DestinationAreaHandler:
    """Factory function to generate appropriate destination handler.
    
    Args:
        config: Configuration object
        world: The world object containing map information
        
    Returns:
        DestinationAreaHandler: Appropriate handler for the destination matrix type
    """
    if hasattr(world.map, 'destination_area_vector') and world.map.destination_area_vector is not None:
        handler = TensorDestinationHandler(config, world)
        logger.info("Using TensorDestinationHandler for destination areas")
    else:
        handler = StandardDestinationHandler(config, world)
        logger.info("Using StandardDestinationHandler for destination areas")
    
    return handler
