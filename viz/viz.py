from typing import Dict, List, Tuple, Union
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot_surface(x: np.array, y: np.array, z: np.array, title: str) -> None:
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                              cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    ax.set_yticks(range(1, 11))
    ax.set_xticks(range(12, 22))
    fig.colorbar(surface)
    plt.show()


def plot_value_function(v: Dict[Tuple, float],
                        title: str = 'Value Function') -> None:
    min_x = min(k[0] for k in v.keys())
    max_x = max(k[0] for k in v.keys())
    min_y = min(k[1] for k in v.keys())
    max_y = max(k[1] for k in v.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)

    x, y = np.meshgrid(x_range, y_range)

    z_no_ace = np.apply_along_axis(lambda _: v[(_[0], _[1], False)], 2, np.dstack([x, y]))
    z_ace = np.apply_along_axis(lambda _: v[(_[0], _[1], True)], 2, np.dstack([x, y]))

    plot_surface(x, y, z_no_ace, '{} (No Usable Ace)'.format(title))
    plot_surface(x, y, z_ace, '{} (Usable Ace)'.format(title))
