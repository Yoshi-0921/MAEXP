# -*- coding: utf-8 -*-

"""Source code for simple environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from .abstract_map import AbstractMap


class SimpleMap(AbstractMap):
    def set_objects_area(self):
        if self.config.type_objects == 1:
            # Set objects area for object 0
            self.objects_area_matrix[0, 1:self.config.map.SIZE_X - 1, 1:self.config.map.SIZE_Y - 1] = 1

        elif self.config.type_objects == 2:
            # Set objects area for object 0
            self.objects_area_matrix[0, 1:self.config.map.SIZE_X - 1, 1:self.config.map.SIZE_Y - 1] = 1
            # Set objects area for object 1
            self.objects_area_matrix[0, 1:self.config.map.SIZE_X - 1, 1:self.config.map.SIZE_Y - 1] = 1
