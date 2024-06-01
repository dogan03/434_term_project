import math
import time

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean


def distanceBetweenPoints(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points in a 2D plane.

    Args:
    x1, y1: Coordinates of the first point
    x2, y2: Coordinates of the second point

    Returns:
    Distance between the two points
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def interpolateWalls(walls_pos_list, num_points=5):
    start_time = time.time()
    interpolated_walls = []
    for idx_1, pos_1 in enumerate(walls_pos_list):
        for idx_2, pos_2 in enumerate(walls_pos_list):
            if (
                idx_1 != idx_2
                and distanceBetweenPoints(pos_1[0], pos_1[1], pos_2[0], pos_2[1]) == 2
            ):
                if pos_1[0] == pos_2[0]:
                    middle_point = (pos_1[1] + pos_2[1]) / 2
                    interpolated_walls.extend([pos_1, pos_2, [pos_1[0], middle_point]])
                    if num_points > 1:
                        step = (pos_2[1] - pos_1[1]) / (num_points + 1)
                        for i in range(1, num_points + 1):
                            interpolated_walls.append([pos_1[0], pos_1[1] + i * step])
                elif pos_1[1] == pos_2[1]:
                    middle_point = (pos_1[0] + pos_2[0]) / 2
                    interpolated_walls.extend([pos_1, pos_2, [middle_point, pos_1[1]]])
                    if num_points > 1:
                        step = (pos_2[0] - pos_1[0]) / (num_points + 1)
                        for i in range(1, num_points + 1):
                            interpolated_walls.append([pos_1[0] + i * step, pos_1[1]])
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Interpolation took {execution_time} seconds...")
    return interpolated_walls


def quaternion_to_yaw(quaternion):
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]
    yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    return yaw


def is_inside_circle(pos, center, radius):
    x, y = pos
    center_x, center_y = center
    distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2
    return distance_squared <= radius**2


def inside_circle(positions, center, radius):
    res = []
    for pos in positions:
        if is_inside_circle(pos, center, radius):
            res.append(pos)

    return res
