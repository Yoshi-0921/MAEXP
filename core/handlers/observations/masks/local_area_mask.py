# -*- coding: utf-8 -*-

"""Returns local observation mask.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from core.worlds.abstract_world import AbstractWorld

from .abstract_observation_coordinate_handler import \
    AbstractObservationCoordinateHandler


class LocalObservationCoordinateHandler(AbstractObservationCoordinateHandler):
    def get_mask_coordinates(self, agent):
        pos_x, pos_y = self.world.map.coord2ind(agent.xy)
        obs_x_min = abs(min(0, pos_x - self.visible_radius))
        obs_x_max = self.visible_range - max(
            0, pos_x + self.visible_radius - (self.world.map.SIZE_X - 1)
        )
        obs_y_min = abs(min(0, pos_y - self.visible_radius))
        obs_y_max = self.visible_range - max(
            0, pos_y + self.visible_radius - (self.world.map.SIZE_Y - 1)
        )
        map_x_min = max(0, pos_x - self.visible_radius)
        map_x_max = min(self.world.map.SIZE_X - 1, pos_x + self.visible_radius) + 1
        map_y_min = max(0, pos_y - self.visible_radius)
        map_y_max = min(self.world.map.SIZE_Y - 1, pos_y + self.visible_radius) + 1

        return {
            "obs_x_min": obs_x_min,
            "obs_x_max": obs_x_max,
            "obs_y_min": obs_y_min,
            "obs_y_max": obs_y_max,
            "area_x_min": obs_x_min,
            "area_x_max": obs_x_max,
            "area_y_min": obs_y_min,
            "area_y_max": obs_y_max,
            "map_x_min": map_x_min,
            "map_x_max": map_x_max,
            "map_y_min": map_y_min,
            "map_y_max": map_y_max,
        }


def generate_local_area_mask(
    config: DictConfig, world: AbstractWorld
) -> torch.Tensor:
    mask = np.zeros(
        (
            world.map.SIZE_X,
            world.map.SIZE_Y,
            config.visible_range,
            config.visible_range,
        ),
        dtype=np.int8,
    ) - 1
    visible_range = config.visible_range
    visible_radius = config.visible_range // 2
    with tqdm(total=world.map.SIZE_X - 2) as pbar:
        pbar.set_description("Generating mask")
        for x in range(1, world.map.SIZE_X - 1):
            for y in range(1, world.map.SIZE_Y - 1):

                if config.transparent_observation:
                    obs_x_min = abs(min(0, x - visible_radius))
                    obs_x_max = visible_range - max(
                        0, x + visible_radius - (world.map.SIZE_X - 1)
                    )
                    obs_y_min = abs(min(0, y - visible_radius))
                    obs_y_max = visible_range - max(
                        0, y + visible_radius - (world.map.SIZE_Y - 1)
                    )
                    global_x_min = max(0, x - visible_radius)
                    global_x_max = min(world.map.SIZE_X - 1, x + visible_radius) + 1
                    global_y_min = max(0, y - visible_radius)
                    global_y_max = min(world.map.SIZE_Y - 1, y + visible_radius) + 1

                    mask[
                        x, y,
                        obs_x_min: obs_x_max,
                        obs_y_min: obs_y_max,
                    ] = world.map.wall_matrix[
                        global_x_min: global_x_max,
                        global_y_min: global_y_max,
                    ] * -1

                else:
                    mask[x, y] = generate_each_mask(x, y, world.map, config.visible_range)
            pbar.update(1)
        pbar.close()

    return mask + 1


def generate_each_mask(x, y, world_map, visible_range) -> np.ndarray:
    visible_radius = visible_range // 2
    mask = np.zeros((visible_range, visible_range), dtype=np.int8)
    mask -= 1
    mask[visible_radius, visible_radius] = 0
    x_coord, y_coord = world_map.ind2coord((x, y))

    for dx in range(-1, 2):
        for dy in [-1, 1]:
            for opr in [-1, 1]:
                for j in range(visible_radius):
                    pos_x, pos_y = dx + j * opr, dy + j * dy
                    local_pos_x, local_pos_y = world_map.coord2ind(
                        (pos_x, pos_y), visible_range, visible_range
                    )
                    pos_x, pos_y = world_map.coord2ind(
                        (
                            pos_x + x_coord,
                            pos_y + y_coord,
                        )
                    )
                    # 場外なら-1
                    if (
                        pos_x < 0
                        or world_map.SIZE_X <= pos_x
                        or pos_y < 0
                        or world_map.SIZE_Y <= pos_y
                    ):
                        mask[local_pos_x, local_pos_y] = -1
                        break
                    # セルが真ん中で壁のない角の方向なら続ける
                    if (
                        j == 0
                        and dx == 0
                        and world_map.wall_matrix[pos_x + opr, pos_y] == 0
                    ):
                        mask[local_pos_x, local_pos_y] = 0
                        continue
                    # セルが角で壁のない真ん中の方向なら続ける、あるならbreak
                    if j == 0 and dx != 0 and opr != dx:
                        if world_map.wall_matrix[pos_x - dx, pos_y] == 0:
                            continue
                        elif world_map.wall_matrix[pos_x - dx, pos_y] == 1:
                            break
                    # 壁なら-1
                    if world_map.wall_matrix[pos_x, pos_y] == 1:
                        mask[local_pos_x, local_pos_y] = -1
                        break
                    mask[local_pos_x, local_pos_y] = 0

    for dy in range(-1, 2):
        for dx in [-1, 1]:
            for opr in [-1, 1]:
                for j in range(visible_radius):
                    pos_x, pos_y = dx + j * dx, dy + j * opr
                    local_pos_x, local_pos_y = world_map.coord2ind(
                        (pos_x, pos_y), visible_range, visible_range
                    )
                    pos_x, pos_y = world_map.coord2ind(
                        (
                            pos_x + x_coord,
                            pos_y + y_coord,
                        )
                    )
                    # 場外なら-1
                    if (
                        pos_x < 0
                        or world_map.SIZE_X <= pos_x
                        or pos_y < 0
                        or world_map.SIZE_Y <= pos_y
                    ):
                        mask[local_pos_x, local_pos_y] = -1
                        break
                    # セルが真ん中で壁のない角の方向なら続ける
                    if (
                        j == 0
                        and dy == 0
                        and world_map.wall_matrix[pos_x, pos_y - opr] == 0
                    ):
                        mask[local_pos_x, local_pos_y] = 0
                        continue
                    # セルが角で壁のない真ん中の方向なら続ける、あるならbreak
                    if j == 0 and dy != 0 and opr != dy:
                        if world_map.wall_matrix[pos_x, pos_y + dy] == 0:
                            continue
                        elif world_map.wall_matrix[pos_x, pos_y + dy] == 1:
                            break
                    # 壁なら-1
                    if world_map.wall_matrix[pos_x, pos_y] == 1:
                        mask[local_pos_x, local_pos_y] = -1
                        break
                    mask[local_pos_x, local_pos_y] = 0

    for opr_x in [-1, 1]:
        for dx in range(visible_radius + 1):
            # 壁ならbreak
            pos_x, pos_y = (
                x_coord + (dx * opr_x),
                y_coord,
            )
            pos_x, pos_y = world_map.coord2ind((pos_x, pos_y))
            local_pos_x, local_pos_y = world_map.coord2ind(
                ((dx * opr_x), 0), visible_range, visible_range
            )
            if world_map.wall_matrix[pos_x, pos_y] == 1:
                mask[local_pos_x, local_pos_y] = -1
                break
            for opr_y in [-1, 1]:
                for dy in range(2):
                    pos_x, pos_y = (
                        x_coord + (dx * opr_x),
                        y_coord + (dy * opr_y),
                    )
                    pos_x, pos_y = world_map.coord2ind((pos_x, pos_y))
                    local_pos_x, local_pos_y = world_map.coord2ind(
                        ((dx * opr_x), (dy * opr_y)),
                        visible_range,
                        visible_range,
                    )
                    # 場外なら-1
                    if (
                        pos_x < 0
                        or world_map.SIZE_X <= pos_x
                        or pos_y < 0
                        or world_map.SIZE_Y <= pos_y
                    ):
                        mask[local_pos_x, local_pos_y] = -1
                        continue
                    # 壁なら-1
                    if world_map.wall_matrix[pos_x, pos_y] == 1:
                        mask[local_pos_x, local_pos_y] = -1
                        break
                    # 何もないなら0
                    else:
                        mask[local_pos_x, local_pos_y] = 0

        for opr_y in [-1, 1]:
            for dx in range(visible_radius + 1):
                # 壁ならbreak
                pos_x, pos_y = (
                    x_coord + (dx * opr_x),
                    y_coord + opr_y,
                )
                pos_x, pos_y = world_map.coord2ind((pos_x, pos_y))
                local_pos_x, local_pos_y = world_map.coord2ind(
                    ((dx * opr_x), opr_y), visible_range, visible_range
                )
                if (
                    pos_x < 0
                    or world_map.SIZE_X <= pos_x
                    or pos_y < 0
                    or world_map.SIZE_Y <= pos_y
                ):
                    mask[local_pos_x, local_pos_y] = -1
                    continue
                # 壁なら-1
                if world_map.wall_matrix[pos_x, pos_y] == 1:
                    mask[local_pos_x, local_pos_y] = -1
                    break
                # 何もないなら0
                else:
                    mask[local_pos_x, local_pos_y] = 0

    for opr_y in [-1, 1]:
        for dy in range(visible_radius + 1):
            # 壁ならbreak
            pos_x, pos_y = x_coord, y_coord + (dy * opr_y)
            pos_x, pos_y = world_map.coord2ind((pos_x, pos_y))
            local_pos_x, local_pos_y = world_map.coord2ind(
                (0, (dy * opr_y)), visible_range, visible_range
            )
            if world_map.wall_matrix[pos_x, pos_y] == 1:
                mask[local_pos_x, local_pos_y] = -1
                break
            for opr_x in [-1, 1]:
                for dx in range(2):
                    pos_x, pos_y = (
                        x_coord + (dx * opr_x),
                        y_coord + (dy * opr_y),
                    )
                    pos_x, pos_y = world_map.coord2ind((pos_x, pos_y))
                    local_pos_x, local_pos_y = world_map.coord2ind(
                        ((dx * opr_x), (dy * opr_y)),
                        visible_range,
                        visible_range,
                    )
                    # 場外なら-1
                    if (
                        pos_x < 0
                        or world_map.SIZE_X <= pos_x
                        or pos_y < 0
                        or world_map.SIZE_Y <= pos_y
                    ):
                        mask[local_pos_x, local_pos_y] = -1
                        continue
                    # 壁なら-1
                    if world_map.wall_matrix[pos_x, pos_y] == 1:
                        mask[local_pos_x, local_pos_y] = -1
                        break
                    # 何もないなら0
                    else:
                        mask[local_pos_x, local_pos_y] = 0

        for opr_x in [-1, 1]:
            for dy in range(visible_radius + 1):
                pos_x, pos_y = (
                    x_coord + opr_x,
                    y_coord + (dy * opr_y),
                )
                pos_x, pos_y = world_map.coord2ind((pos_x, pos_y))
                local_pos_x, local_pos_y = world_map.coord2ind(
                    (opr_x, (dy * opr_y)),
                    visible_range,
                    visible_range,
                )
                # 場外なら-1
                if (
                    pos_x < 0
                    or world_map.SIZE_X <= pos_x
                    or pos_y < 0
                    or world_map.SIZE_Y <= pos_y
                ):
                    mask[local_pos_x, local_pos_y] = -1
                    continue
                # 壁なら-1
                if world_map.wall_matrix[pos_x, pos_y] == 1:
                    mask[local_pos_x, local_pos_y] = -1
                    break
                # 何もないなら0
                else:
                    mask[local_pos_x, local_pos_y] = 0

    return mask
