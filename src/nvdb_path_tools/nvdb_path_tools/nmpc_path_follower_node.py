import rclpy
from rclpy.node import Node

import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, Bool

from nvdb_path_tools.nmpc_core import (
    NMPCController,
    VehicleParams,
    ControllerParams,
    ModelParams,
    SpeedProfileParams,
    SplinePath,
    build_curvature_aware_speed_profile,
    preview_path_data,
    wrap_angle,
)


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)


class NMPCPathFollowerNode(Node):

    def __init__(self):
        super().__init__("nmpc_path_follower")

        self.create_subscription(PoseStamped, "/pose", self.pose_callback, 10)
        self.create_subscription(Path, "/path", self.path_callback, 10)
        self.create_subscription(Float64, "/estimated_car_speed", self.speed_callback, 10)
        self.create_subscription(Float64, "/steering_wheel_angle_filtered", self.steering_measurement_callback, 10)

        self.steering_pub = self.create_publisher(Float64, "/steering_angle_stanley", 10)
        self.speed_pub = self.create_publisher(Float64, "/cruise_control_setpoint_speed", 10)
        self.status_pub = self.create_publisher(Bool, "/path_following_status", 10)

        self.pred_path_pub = self.create_publisher(Path, "/nmpc_prediction", 10)
        self.next_pose_pub = self.create_publisher(PoseStamped, "/nmpc_next_pose", 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.x = None
        self.y = None
        self.yaw = None

        self.delta = 0.0
        self.vy = 0.0
        self.r = 0.0
        self.vx = 7.0
        self.last_s = 0.0
        self.last_valid_delta_ref = 0.0

        self.last_path_num_points = None
        self.last_path_length_estimate = None

        self.MAX_STEERING_ANGLE = np.arcsin(2.72 / (5.3 - 0.215 / 2.0))
        self.MAX_STEERING_WHEEL_ANGLE = 460.0 * np.pi / 180.0
        self.desired_speed_kmh = 8.0 * 3.6

        self.ctrl_par = ControllerParams()
        self.speed_par = SpeedProfileParams()

        self.model_ctrl = ModelParams(
            tire_model="dugoff",
            denom_eps=1e-3,
            vx_eps=0.5,
            traction_front_frac=0.6,
            braking_front_frac=0.7,
        )

        self.veh_ctrl = VehicleParams(
            m=1757.0,
            Iz=3100.0,
            lf=1.23,
            lr=1.49,
            Cf=125000.0,
            Cr=118000.0,
            mu=0.80,
        )

        self.nmpc_path = None
        self.speed_profile = None
        self.ax_profile = None
        self.controller = None

        self.create_fallback_path()

        self.get_logger().info(
            "NMPC path follower initialized with dynamic bicycle model and Dugoff tire model."
        )

    def create_fallback_path(self):
        x_wp = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
        y_wp = np.array([0, 0, 0, 1, 3, 6, 8, 9, 9, 9], dtype=float)
        self.set_path_from_waypoints(x_wp, y_wp, source_name="fallback path")

    def reset_solver_warm_start(self):
        if self.controller is not None:
            self.controller.last_sol = None
            self.controller.last_u0 = np.array([0.0], dtype=float)

    def set_path_from_waypoints(self, x_wp, y_wp, source_name="ROS path"):
        self.nmpc_path = SplinePath(x_wp, y_wp, ds=0.5)

        self.speed_profile = build_curvature_aware_speed_profile(
            self.nmpc_path,
            self.speed_par,
        )

        self.ax_profile = np.zeros_like(self.speed_profile)

        self.controller = NMPCController(
            self.nmpc_path,
            self.veh_ctrl,
            self.ctrl_par,
            self.model_ctrl,
        )

        self.last_s = 0.0
        self.delta = 0.0
        self.last_valid_delta_ref = 0.0
        self.reset_solver_warm_start()

        self.get_logger().info(
            f"Loaded {source_name}: {len(x_wp)} waypoints, length={self.nmpc_path.length:.2f} m"
        )

    def pose_callback(self, msg):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.yaw = yaw_from_quaternion(msg.pose.orientation)

    def speed_callback(self, msg):
        self.vx = max(float(msg.data) / 3.6, self.model_ctrl.vx_eps)

    def steering_measurement_callback(self, msg):
        steering_wheel_deg = float(msg.data)

        self.delta = -steering_wheel_deg * (30.0 / 460.0) * np.pi / 180.0

        self.delta = float(np.clip(
            self.delta,
            -self.MAX_STEERING_ANGLE,
            self.MAX_STEERING_ANGLE,
        ))

    def path_callback(self, msg):
        if len(msg.poses) < 2:
            self.get_logger().warn("Received /path with fewer than 2 poses. Ignoring.")
            return

        path_num_points = len(msg.poses)

        x_first = msg.poses[0].pose.position.x
        y_first = msg.poses[0].pose.position.y
        x_last = msg.poses[-1].pose.position.x
        y_last = msg.poses[-1].pose.position.y

        path_length_estimate = np.hypot(
            x_last - x_first,
            y_last - y_first,
        )

        if (
            self.last_path_num_points == path_num_points
            and self.last_path_length_estimate is not None
            and abs(self.last_path_length_estimate - path_length_estimate) < 1e-3
        ):
            return

        self.last_path_num_points = path_num_points
        self.last_path_length_estimate = path_length_estimate

        x_wp = np.array([p.pose.position.x for p in msg.poses], dtype=float)
        y_wp = np.array([p.pose.position.y for p in msg.poses], dtype=float)

        keep = [0]
        for i in range(1, len(x_wp)):
            dx = x_wp[i] - x_wp[keep[-1]]
            dy = y_wp[i] - y_wp[keep[-1]]
            if np.hypot(dx, dy) > 1e-3:
                keep.append(i)

        x_wp = x_wp[keep]
        y_wp = y_wp[keep]

        if len(x_wp) < 2:
            self.get_logger().warn("Received /path only had duplicate points. Ignoring.")
            return

        try:
            self.set_path_from_waypoints(x_wp, y_wp, source_name="/path")
        except Exception as err:
            self.get_logger().warn(f"Failed to build SplinePath from /path: {err}")

    def find_closest_s(self, x_car, y_car, last_s=None, search_radius=10.0):
        path = self.nmpc_path

        if path is None:
            return 0.0

        if last_s is None:
            s_candidates = path.s
        else:
            s_min = max(path.s[0], last_s - search_radius)
            s_max = min(path.s[-1], last_s + search_radius)
            mask = (path.s >= s_min) & (path.s <= s_max)
            s_candidates = path.s[mask]

            if len(s_candidates) == 0:
                s_candidates = path.s

        x_ref = np.interp(s_candidates, path.s, path.x)
        y_ref = np.interp(s_candidates, path.s, path.y)

        dist2 = (x_ref - x_car) ** 2 + (y_ref - y_car) ** 2
        idx = int(np.argmin(dist2))

        return float(s_candidates[idx])

    def compute_errors(self, x_car, y_car, yaw, s):
        x_ref = self.nmpc_path.interp_x(s)
        y_ref = self.nmpc_path.interp_y(s)
        psi_ref = self.nmpc_path.interp_psi(s)

        dx = x_car - x_ref
        dy = y_car - y_ref

        ey = -dx * np.sin(psi_ref) + dy * np.cos(psi_ref)
        epsi = wrap_angle(yaw - psi_ref)

        return ey, epsi

    def error_state_to_pose(self, ey, epsi, s):
        x_ref = self.nmpc_path.interp_x(s)
        y_ref = self.nmpc_path.interp_y(s)
        psi_ref = self.nmpc_path.interp_psi(s)

        x_global = x_ref - ey * np.sin(psi_ref)
        y_global = y_ref + ey * np.cos(psi_ref)
        yaw_global = wrap_angle(psi_ref + epsi)

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"

        pose.pose.position.x = float(x_global)
        pose.pose.position.y = float(y_global)
        pose.pose.position.z = 0.0

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = float(np.sin(yaw_global / 2.0))
        pose.pose.orientation.w = float(np.cos(yaw_global / 2.0))

        return pose

    def wheel_angle_to_steering_wheel_deg(self, delta_ref):
        steering_wheel_angle_rad = (
            self.MAX_STEERING_WHEEL_ANGLE / self.MAX_STEERING_ANGLE
        ) * delta_ref

        return float(np.rad2deg(steering_wheel_angle_rad))

    def fallback_delta_ref(self):
        delta_dot_fallback = -self.delta / self.ctrl_par.dt

        if self.controller is not None:
            delta_dot_fallback = np.clip(
                delta_dot_fallback,
                -self.controller.ddelta_max,
                self.controller.ddelta_max,
            )

        delta_ref = self.delta + delta_dot_fallback * self.ctrl_par.dt

        if self.controller is not None:
            delta_ref = np.clip(
                delta_ref,
                -self.controller.delta_max,
                self.controller.delta_max,
            )

        return float(delta_ref)

    def publish_outputs(self, delta_ref):
        steering_msg = Float64()
        steering_msg.data = self.wheel_angle_to_steering_wheel_deg(delta_ref)
        self.steering_pub.publish(steering_msg)

        speed_msg = Float64()
        speed_msg.data = float(self.desired_speed_kmh)
        self.speed_pub.publish(speed_msg)

        status_msg = Bool()
        status_msg.data = True
        self.status_pub.publish(status_msg)

        return steering_msg.data

    def publish_prediction(self, X_opt, s_preview):
        pred_msg = Path()
        pred_msg.header.stamp = self.get_clock().now().to_msg()
        pred_msg.header.frame_id = "map"

        for k in range(X_opt.shape[0]):
            ey_k = float(X_opt[k, 0])
            epsi_k = float(X_opt[k, 1])
            s_k = float(s_preview[min(k, len(s_preview) - 1)])

            pose = self.error_state_to_pose(ey_k, epsi_k, s_k)
            pose.header = pred_msg.header
            pred_msg.poses.append(pose)

        self.pred_path_pub.publish(pred_msg)

        if X_opt.shape[0] > 1:
            ey_next = float(X_opt[1, 0])
            epsi_next = float(X_opt[1, 1])
            s_next = float(s_preview[min(1, len(s_preview) - 1)])

            next_pose = self.error_state_to_pose(ey_next, epsi_next, s_next)
            self.next_pose_pub.publish(next_pose)

    def control_loop(self):
        if self.x is None or self.y is None or self.yaw is None:
            return

        if self.nmpc_path is None or self.controller is None:
            self.get_logger().warn("No valid path/controller yet.")
            return

        s_current_raw = self.find_closest_s(
            self.x,
            self.y,
            last_s=self.last_s,
            search_radius=10.0,
        )

        if abs(s_current_raw - self.last_s) > 5.0:
            self.get_logger().warn(
                f"Large s jump detected: last_s={self.last_s:.2f}, "
                f"new_s={s_current_raw:.2f}. Resetting warm start."
            )
            self.reset_solver_warm_start()

        s_current = s_current_raw

        if s_current < self.last_s - 1.0:
            s_current = self.last_s

        self.last_s = s_current

        ey, epsi = self.compute_errors(
            self.x,
            self.y,
            self.yaw,
            s_current,
        )

        x0 = np.array([
            ey,
            epsi,
            self.vy,
            self.r,
            self.delta,
        ], dtype=float)

        s_preview, kappa_preview, vx_preview, ax_preview = preview_path_data(
            self.nmpc_path,
            self.speed_profile,
            self.ax_profile,
            x0,
            s_current,
            self.ctrl_par,
            self.model_ctrl,
        )

        vx_preview = np.ones_like(vx_preview) * max(self.vx, self.model_ctrl.vx_eps)

        delta_ref = self.delta
        solve_status = "NOT_SOLVED"

        try:
            X_opt, U_opt, solve_time, iter_count, status, soft_failure = self.controller.solve(
                x0,
                kappa_preview,
                vx_preview,
                ax_preview,
            )

            self.publish_prediction(X_opt, s_preview)

            delta_dot = float(U_opt[0, 0])
            delta_ref = self.delta + delta_dot * self.ctrl_par.dt

            delta_ref = np.clip(
                delta_ref,
                -self.controller.delta_max,
                self.controller.delta_max,
            )

            delta_ref = float(delta_ref)
            self.last_valid_delta_ref = delta_ref
            solve_status = status

        except Exception as err:
            self.get_logger().warn(
                f"NMPC failed. Resetting warm start and using fallback steering. Error: {err}"
            )

            self.reset_solver_warm_start()
            delta_ref = self.fallback_delta_ref()
            solve_status = "FALLBACK"

        steering_wheel_deg = self.publish_outputs(delta_ref)

        self.get_logger().info(
            f"s={s_current:.2f}, ey={ey:.3f}, "
            f"epsi={np.rad2deg(epsi):.2f} deg, "
            f"vx={self.vx:.2f} m/s, "
            f"delta_meas={np.rad2deg(self.delta):.2f} deg, "
            f"wheel_delta_ref={np.rad2deg(delta_ref):.2f} deg, "
            f"steering_wheel_ref={steering_wheel_deg:.2f} deg, "
            f"tire_model=dugoff, "
            f"status={solve_status}"
        )


def main():
    rclpy.init()
    node = NMPCPathFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()