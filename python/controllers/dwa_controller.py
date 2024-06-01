import math
import time
from enum import Enum

import numpy as np


class RobotType(Enum):
    circle = 0
    rectangle = 1


class DWAController:
    def __init__(self, config) -> None:
        self.config = config

    def dwa_control(self, x, goal, ob):
        """
        Dynamic Window Approach control
        """
        dw = self.calc_dynamic_window(x)

        u, trajectory, min_r, to_goal_cost = self.calc_control_and_trajectory(
            x, dw, goal, ob
        )

        return u, trajectory, min_r, to_goal_cost

    def calc_dynamic_window(self, x):
        """
        calculation dynamic window based on current state x
        """

        Vs = [
            self.config.min_speed,
            self.config.max_speed,
            -self.config.max_yaw_rate,
            self.config.max_yaw_rate,
        ]

        Vd = [
            x[3] - self.config.max_accel * self.config.dt,
            x[3] + self.config.max_accel * self.config.dt,
            x[4] - self.config.max_delta_yaw_rate * self.config.dt,
            x[4] + self.config.max_delta_yaw_rate * self.config.dt,
        ]

        dw = [
            max(Vs[0], Vd[0]),
            min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]),
            min(Vs[3], Vd[3]),
        ]

        return dw

    def predict_trajectory(self, x_init, v, y):
        """
        predict trajectory with an input
        """

        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.config.predict_time:
            x = self.motion(x, [v, y])
            trajectory = np.vstack((trajectory, x))
            time += self.config.dt
        return trajectory

    def calc_control_and_trajectory(self, x, dw, goal, ob):
        """
        calculation final input with dynamic window
        """
        start = time.time()

        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        for y in np.arange(dw[2], dw[3], self.config.yaw_rate_resolution):

            trajectory = self.predict_trajectory(x_init, 2, y)
            # calc cost
            to_goal_cost = self.config.to_goal_cost_gain * self.calc_to_goal_cost(
                trajectory, goal
            )
            speed_cost = self.config.speed_cost_gain * (
                self.config.max_speed - trajectory[-1, 3]
            )
            obstacle_cost_return = self.calc_obstacle_cost(trajectory, ob, self.config)
            ob_cost = self.config.obstacle_cost_gain * obstacle_cost_return[0]
            min_r = obstacle_cost_return[1]

            final_cost = to_goal_cost + speed_cost + ob_cost

            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [2, y]
                best_trajectory = trajectory
                if (
                    abs(best_u[0]) < self.config.robot_stuck_flag_cons
                    and abs(x[3]) < self.config.robot_stuck_flag_cons
                ):
                    best_u[1] = -self.config.max_delta_yaw_rate
        end = time.time()
        exec_cost = end - start
        return best_u, best_trajectory, min_r, to_goal_cost

    def calc_obstacle_cost(self, trajectory, ob, config):
        """
        calc obstacle cost inf: collision
        """
        ox = ob[:, 0]
        oy = ob[:, 1]
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)

        if self.config.robot_type == RobotType.rectangle:
            yaw = trajectory[:, 2]
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rot = np.transpose(rot, [2, 0, 1])
            local_ob = ob[:, None] - trajectory[:, 0:2]
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            local_ob = np.array([local_ob @ x for x in rot])
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            upper_check = local_ob[:, 0] <= config.robot_length / 2
            right_check = local_ob[:, 1] <= config.robot_width / 2
            bottom_check = local_ob[:, 0] >= -config.robot_length / 2
            left_check = local_ob[:, 1] >= -config.robot_width / 2
            if (
                np.logical_and(
                    np.logical_and(upper_check, right_check),
                    np.logical_and(bottom_check, left_check),
                )
            ).any():
                return float("Inf")
        elif self.config.robot_type == RobotType.circle:
            if np.array(r <= config.robot_radius).any():
                return float("Inf")

        min_r = np.min(r) - 0.55
        return 1.0 / min_r, min_r

    def calc_to_goal_cost(self, trajectory, goal):
        """
        calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost

    def motion(self, x, u):
        """
        motion model
        """

        x[2] += u[1] * self.config.dt
        x[0] += u[0] * math.cos(x[2]) * self.config.dt
        x[1] += u[0] * math.sin(x[2]) * self.config.dt
        x[3] = u[0]
        x[4] = u[1]

        return x
