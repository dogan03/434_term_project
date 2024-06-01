import random
import time
from enum import Enum

import cmpe434_dungeon
import cmpe434_utils
import mujoco
import mujoco.viewer
import numpy as np
import scipy as sp
from controllers.dwa_controller import DWAController
from planner import AStarPlanner
from plotter import *
from utils import *


def key_callback(keycode):
    if chr(keycode) == " ":
        global paused
        paused = not paused


paused = False


def create_scenario():
    tile_idx = 0

    scene, scene_assets = cmpe434_utils.get_model("scenes/empty_floor.xml")
    tiles, rooms, connections = cmpe434_dungeon.generate(3, 2, 8)

    for index, r in enumerate(rooms):
        (xmin, ymin, xmax, ymax) = cmpe434_dungeon.find_room_corners(r)
        scene.worldbody.add(
            "geom",
            name="R{}".format(index),
            type="plane",
            size=[(xmax - xmin) + 1, (ymax - ymin) + 1, 1],
            rgba=[0, 0, 0, 0],
            pos=[(xmin + xmax), (ymin + ymax), 0.0001],
        )

    for pos, tile in tiles.items():
        if tile == "#":
            scene.worldbody.add(
                "geom",
                type="box",
                name="T{}".format(tile_idx),
                size=[1, 1, 0.1],
                rgba=[0.8, 0.6, 0.4, 1],
                pos=[pos[0] * 2, pos[1] * 2, 0],
            )
            tile_idx += 1

    robot, robot_assets = cmpe434_utils.get_model("models/mushr_car/model.xml")
    start_pos = random.choice([key for key in tiles.keys() if tiles[key] == "."])
    final_pos = random.choice(
        [key for key in tiles.keys() if tiles[key] == "." and key != start_pos]
    )

    scene.worldbody.add(
        "site",
        name="start",
        type="box",
        size=[0.5, 0.5, 0.01],
        rgba=[0, 0, 1, 1],
        pos=[start_pos[0] * 2, start_pos[1] * 2, 0],
    )
    start_position = [start_pos[0] * 2, start_pos[1] * 2, 0]
    scene.worldbody.add(
        "site",
        name="finish",
        type="box",
        size=[0.5, 0.5, 0.01],
        rgba=[1, 0, 0, 1],
        pos=[final_pos[0] * 2, final_pos[1] * 2, 0],
    )
    final_position = [final_pos[0] * 2, final_pos[1] * 2, 0]
    for i, room in enumerate(rooms):
        obs_pos = random.choice(
            [tile for tile in room if tile != start_pos and tile != final_pos]
        )
        scene.worldbody.add(
            "geom",
            name="Z{}".format(i),
            type="cylinder",
            size=[0.2, 0.05],
            rgba=[0.8, 0.0, 0.1, 1],
            pos=[obs_pos[0] * 2, obs_pos[1] * 2, 0.08],
        )

    start_yaw = random.randint(0, 359)
    robot.find("body", "buddy").set_attributes(
        pos=[start_pos[0] * 2, start_pos[1] * 2, 0.1], euler=[0, 0, start_yaw]
    )

    scene.include_copy(robot)

    all_assets = {**scene_assets, **robot_assets}

    return scene, all_assets, start_position, final_position


def get_square_corners(centers, side_length):
    """
    Calculate the coordinates of the corners of squares given their centers and side length.

    Parameters:
        centers (list of tuples): List of center coordinates of the squares.
        side_length (float): Side length of the squares.

    Returns:
        list of lists: List of lists where each sublist contains the coordinates of the corners of a square.
    """
    corners_list = []
    for center in centers:
        x_center, y_center = center
        x_corner = x_center - side_length / 2
        y_corner = y_center - side_length / 2
        corners = [
            (x_corner, y_corner),
            (x_corner + side_length, y_corner),
            (x_corner + side_length, y_corner + side_length),
            (x_corner, y_corner + side_length),
        ]
        corners_list.append(corners)
    crnrs = []
    for corners in corners_list:
        crnrs.extend(corners)
    return crnrs


def execute_scenario(scene, start_pos, final_pos, ASSETS=dict()):
    start_t = 0
    saved_idx = 0
    radius_treshold = 1.2
    all_obstacles = []

    init_dwa = False
    goal_at_final = False
    add_pursuit_track = True

    m = mujoco.MjModel.from_xml_string(scene.to_xml_string(), assets=all_assets)
    d = mujoco.MjData(m)
    rooms = [m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("R")]
    obstacles = [m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("Z")]

    uniform_direction_dist = sp.stats.uniform_direction(2)
    obstacle_direction = [
        [x, y, 0] for x, y in uniform_direction_dist.rvs(len(obstacles))
    ]

    unused = np.zeros(1, dtype=np.int32)

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        velocity = d.actuator("throttle_velocity")
        steering = d.actuator("steering")

        start = time.time()
        walls_pos = []
        obs_positions = []

        for i in range(m.ngeom):
            if m.geom(i).name.startswith("T"):
                walls_pos.append(m.geom(i).pos[:2])
        for i in range(m.ngeom):
            if m.geom(i).name.startswith("Z"):
                obs_positions.append(m.geom(i).pos[:2])
        corners = get_square_corners(walls_pos, 1)

        walls_x = [x for x, _ in corners]
        walls_y = [y for _, y in corners]
        all_obstacles.extend(corners)

        class RobotType(Enum):
            circle = 0
            rectangle = 1

        class Config:
            def __init__(self):
                self.max_speed = 6.0  # [m/s]
                self.min_speed = -3.0  # [m/s]
                self.max_yaw_rate = 50 * (40.0 * math.pi / 180.0)  # [rad/s]
                self.max_accel = 0.2  # [m/ss]
                self.max_delta_yaw_rate = 6 * 1 * (40.0 * math.pi / 180.0)  # [rad/ss]
                self.v_resolution = 0.1  # [m/s]
                self.yaw_rate_resolution = 1 * math.pi / 180.0  # [rad/s]
                self.dt = 0.1  # [s] Time tick for motion prediction
                self.predict_time = m.opt.timestep  # [s]
                self.to_goal_cost_gain = 1
                self.speed_cost_gain = 0
                self.obstacle_cost_gain = 3  # it was 10
                self.robot_stuck_flag_cons = 0.001
                self.robot_type = RobotType.circle

                # if robot_type == RobotType.circle
                # Also used to check if goal is reached in both types
                self.robot_radius = 1  # [m] for collision check

                # if robot_type == RobotType.rectangle
                self.robot_width = 3.0  # [m] for collision check
                self.robot_length = 6.4  # [m] for collision check

            @property
            def robot_type(self):
                return self._robot_type

            @robot_type.setter
            def robot_type(self, value):
                if not isinstance(value, RobotType):
                    raise TypeError("robot_type must be an instance of RobotType")
                self._robot_type = value

        config = Config()

        dwa_controller = DWAController(config)

        grid_size = 1
        robot_radius = 1.5
        planner = AStarPlanner(walls_x, walls_y, grid_size, robot_radius)
        rx, ry = planner.planning(
            start_pos[0], start_pos[1], final_pos[0], final_pos[1]
        )
        rx, ry = rx[::-1], ry[::-1]
        print("PATH PLANNED")

        velocity.ctrl = 0.0
        steering.ctrl = 0.0
        while viewer.is_running() and time.time() - start < 300:
            step_start = time.time()

            if not paused:
                robot_x_pos = d.xpos[1][0]
                robot_y_pos = d.xpos[1][1]
                all_obstacles_copy = all_obstacles.copy()
                next_obstacle_positions = []
                for i, x in enumerate(obstacles):
                    dx = obstacle_direction[i][0]
                    dy = obstacle_direction[i][1]

                    px = m.geom_pos[x][0]
                    py = m.geom_pos[x][1]
                    pz = 0.02

                    nearest_dist = mujoco.mj_ray(
                        m, d, [px, py, pz], obstacle_direction[i], None, 1, -1, unused
                    )

                    if nearest_dist >= 0 and nearest_dist < 0.4:
                        obstacle_direction[i][0] = -dy
                        obstacle_direction[i][1] = dx

                    m.geom_pos[x][0] = m.geom_pos[x][0] + dx * 0.001
                    m.geom_pos[x][1] = m.geom_pos[x][1] + dy * 0.001
                    next_x = m.geom_pos[x][0] + dx * 0.001
                    next_y = m.geom_pos[x][1] + dy * 0.001
                    next_obstacle_positions.append([next_x, next_y])
                all_obstacles_copy.extend(next_obstacle_positions)
                if not init_dwa:
                    x = [
                        d.xpos[1][0],
                        d.xpos[1][1],
                        quaternion_to_yaw(d.xquat[1]),
                        velocity.ctrl[0],
                        steering.ctrl[0],
                    ]
                    init_dwa = True
                else:
                    for i in range(saved_idx, len(rx)):
                        if 1 <= distanceBetweenPoints(
                            d.xpos[1][0], d.xpos[1][1], rx[i], ry[i]
                        ):
                            goal_x = rx[i]
                            goal_y = ry[i]
                            saved_idx = i
                            break
                        elif i == len(rx) - 1:
                            goal_x = rx[i]
                            goal_y = ry[i]
                            goal_at_final = True

                    if start_t % 3 == 0:
                        x = [
                            d.xpos[1][0],
                            d.xpos[1][1],
                            quaternion_to_yaw(d.xquat[1]),
                            velocity.ctrl[0],
                            steering.ctrl[0],
                        ]

                        u, trajectory, min_r, to_goal_cost = dwa_controller.dwa_control(
                            x=x,
                            goal=np.array([goal_x, goal_y]),
                            ob=np.array(all_obstacles_copy),
                        )
                        if add_pursuit_track:
                            viewer.user_scn.ngeom = 0

                            mujoco.mjv_initGeom(
                                viewer.user_scn.geoms[0],
                                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[0.1, 0.1, 0],
                                pos=[goal_x, goal_y, 0.2],
                                mat=np.eye(3).flatten(),
                                rgba=[1, 0, 1, 1],
                            )

                            mujoco.mjv_initGeom(
                                viewer.user_scn.geoms[1],
                                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                size=[
                                    radius_treshold,
                                    0.001,
                                    0.01,
                                ],
                                pos=[
                                    robot_x_pos,
                                    robot_y_pos,
                                    0,
                                ],
                                mat=np.eye(3).flatten(),
                                rgba=[1, 1, 0, 1],
                            )

                            viewer.user_scn.ngeom = 2
                        if (
                            goal_at_final
                            and distanceBetweenPoints(
                                robot_x_pos, robot_y_pos, goal_x, goal_y
                            )
                            < 0.4
                        ):
                            return
                        else:
                            if min_r > radius_treshold:
                                if to_goal_cost > 1.5:
                                    velocity.ctrl = 2
                                    if u[1] < 0:
                                        steering.ctrl = min(-7, u[1])
                                    else:
                                        steering.ctrl = max(7, u[1])
                                elif 0.5 < to_goal_cost < 1.5:
                                    velocity.ctrl = 4
                                    if u[1] < 0:
                                        steering.ctrl = max(-4, u[1])
                                    else:
                                        steering.ctrl = min(4, u[1])

                                else:

                                    velocity.ctrl = 5
                                    if u[1] < 0:
                                        steering.ctrl = max(-3, u[1])
                                    else:
                                        steering.ctrl = min(3, u[1])
                            else:
                                if to_goal_cost > 1.5:
                                    velocity.ctrl = 1
                                    steering.ctrl = u[1]
                                else:
                                    velocity.ctrl = 3
                                    steering.ctrl = u[1]
                    start_t += 1
                mujoco.mj_step(m, d)

                viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    return m, d, error_list


if __name__ == "__main__":
    scene, all_assets, start_p, final_p = create_scenario()
    _, _, error_list = execute_scenario(scene, start_p, final_p, all_assets)
