"""Source code for colorizer.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from collections import defaultdict


RGB_COLORS = defaultdict(
    white=[1, 1, 1],
    black=[0, 0, 0],
    red=[1, 0, 0],
    green=[0, 1, 0],
    blue=[0, 0, 1],
    yellow=[1, 1, 0],
    magenta=[1, 0, 1],
    cyan=[0, 1, 1],
    orange=[1, 0.5, 0],
    chartreuse=[0.5, 1, 0],
    pink=[1, 0, 0.5],
    electric_indigo=[0.5, 0, 1],
    spring_green=[0, 1, 0.5],
    dodger_blue=[0, 0.5, 1]

)
