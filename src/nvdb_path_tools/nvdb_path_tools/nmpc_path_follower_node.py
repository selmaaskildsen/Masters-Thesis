import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist, Vector3Stamped
from std_msgs.msg import Float64, Bool
from car_control.msg import VehicleState

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


def make_float64(value):
    msg = Float64()
    msg.data = float(value)
    return msg


def make_bool(value):
    msg = Bool()
    msg.data = bool(value)
    return msg


class NMPCPathFollowerNode(Node):

    STEERING_RATIO = 15.33
    WHEELBASE = 2.79

    def __init__(self):
        super().__init__("nmpc_path_follower")

        # ---------------- Parameters ----------------
        # Direct mode:
        #   use_external_steering_mpc = False, publish_to_cmd_vel = True
        #   NMPC node converts delta_ref -> torque and publishes /cmd_vel.
        #
        # Cascade mode with Rory's steering_mpc_node:
        #   use_external_steering_mpc = True, publish_to_cmd_vel = False
        #   NMPC node publishes delta_ref to /path_follower/cmd_vel.angular.z.
        #   Rory's steering_mpc_node publishes /cmd_vel.
        self.declare_parameter("use_external_steering_mpc", True)
        self.declare_parameter("publish_to_cmd_vel", False)
        self.declare_parameter("desired_speed_mps", 4.0)
        self.declare_parameter("torque_limit", 0.7)
        self.declare_parameter("kp_speed", 0.3)
        self.declare_parameter("path_csv_file", "")

        self.use_external_steering_mpc = bool(
            self.get_parameter("use_external_steering_mpc").value
        )
        self.publish_to_cmd_vel = bool(self.get_parameter("publish_to_cmd_vel").value)

        if self.use_external_steering_mpc and self.publish_to_cmd_vel:
            self.get_logger().warn(
                "use_external_steering_mpc=True but publish_to_cmd_vel=True. "
                "For safety, the NMPC node will NOT publish direct /cmd_vel in external mode."
            )

        # ---------------- Subscriptions ----------------
        self.create_subscription(PoseStamped, "gnss/pose", self.pose_callback, 10)
        self.create_subscription(Vector3Stamped, "gnss/gyro", self.gyro_callback, 10)
        self.create_subscription(VehicleState, "vehicle/state", self.vehicle_state_callback, 10)
        self.create_subscription(Bool, "enable_path_following", self.enable_callback, 10)

        # Support both possible path topic names.
        self.create_subscription(Path, "path", self.path_callback, 10)
        self.create_subscription(Path, "reference_path", self.path_callback, 10)

        # ---------------- Publishers ----------------
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.debug_cmd_vel_pub = self.create_publisher(Twist, "nmpc/debug_cmd_vel", 10)

        qos_latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.status_pub = self.create_publisher(Bool, "path_following_status", qos_latched)
        self.path_vis_pub = self.create_publisher(Path, "path_visualization", qos_latched)

        self.pub_cte = self.create_publisher(Float64, "lateral_mpc/cte_m", 10)
        self.pub_hdg_err = self.create_publisher(Float64, "lateral_mpc/heading_error_deg", 10)
        self.pub_desired_delta = self.create_publisher(Float64, "lateral_mpc/desired_delta_deg", 10)
        self.pub_actual_delta = self.create_publisher(Float64, "lateral_mpc/actual_delta_deg", 10)
        self.pub_steering_error = self.create_publisher(Float64, "lateral_mpc/steering_error_deg", 10)
        self.pub_delta_ff = self.create_publisher(Float64, "lateral_mpc/feedforward_delta_deg", 10)
        self.pub_torque_cmd = self.create_publisher(Float64, "lateral_mpc/torque_cmd", 10)
        self.pub_torque_raw = self.create_publisher(Float64, "lateral_mpc/torque_cmd_raw", 10)
        self.pub_progress = self.create_publisher(Float64, "lateral_mpc/progress_m", 10)
        self.pub_distance_to_end = self.create_publisher(Float64, "path_follower/distance_to_end_m", 10)

        self.pub_lateral_error = self.create_publisher(Float64, "lateral_error", 10)
        self.pub_heading_error = self.create_publisher(Float64, "heading_error", 10)

        # This is the key output to Rory's steering_mpc_node in external/cascade mode.
        # angular.z = desired front-wheel steering angle [rad].
        # linear.x is kept as desired speed for logging/compatibility, but Rory's node uses its own speed parameter.
        self.pub_pf_cmd_vel = self.create_publisher(Twist, "path_follower/cmd_vel", 10)

        self.pred_path_pub = self.create_publisher(Path, "nmpc_prediction", 10)

        self.pub_solve_time = self.create_publisher(Float64, "nmpc/solve_time_ms", 10)
        self.pub_speed_gate = self.create_publisher(Bool, "nmpc/speed_gate_active", 10)
        self.pub_yaw_rate_source = self.create_publisher(Float64, "nmpc/yaw_rate_source", 10)
        self.pub_stopping_active = self.create_publisher(Bool, "path_follower/stopping_active", 10)
        self.pub_external_mode = self.create_publisher(Bool, "nmpc/use_external_steering_mpc", 10)

        # ---------------- Timer ----------------
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        # ---------------- Path projection parameters ----------------
        self.initial_search_fraction = 0.5
        self.initial_search_max_length = 40.0
        self.s_search_radius = 10.0
        self.max_forward_s_jump = 5.0
        self.max_backward_s_jump = 1.0

        # ---------------- End-of-path / stopping parameters ----------------
        self.mode = "IDLE"  # IDLE, TRACKING, STOPPING
        self.end_slowdown_distance = 8.0
        self.stop_distance = 3.0
        self.vehicle_stopped_speed = 0.2
        self.end_brake_cmd_fast = -0.35
        self.end_brake_cmd_slow = -0.20

        # ---------------- Vehicle state ----------------
        self.x = None
        self.y = None
        self.yaw = None

        self.prev_yaw = None
        self.prev_yaw_time = None
        self.r_from_yaw = 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.r = 0.0

        self.delta = 0.0
        self.delta_rate = 0.0
        self.prev_delta = None
        self.prev_delta_time = None

        self.last_s = 0.0
        self.last_valid_delta_ref = 0.0

        self.gnss_valid = False
        self.gyro_valid = False
        self.vehicle_state_valid = False

        self.last_pose_time = None
        self.last_gyro_time = None
        self.last_vehicle_state_time = None
        self.max_data_age_s = 1.0

        self.active = False
        self.start_time = None

        # ---------------- Robustness parameters ----------------
        self.nmpc_enable_speed_mps = 3.2
        self.nmpc_disable_speed_mps = 2.8
        self.nmpc_speed_ok = False

        self.consecutive_failures = 0
        self.max_consecutive_failures = 5

        self.soft_start_duration = 3.0

        # ---------------- Control parameters ----------------
        self.desired_speed_mps = float(self.get_parameter("desired_speed_mps").value)
        self.kp_speed = float(self.get_parameter("kp_speed").value)
        self.torque_limit = float(self.get_parameter("torque_limit").value)

        self.ctrl_par = ControllerParams()
        self.speed_par = SpeedProfileParams()

        # Recommended after first full-scale test: make NMPC steering demand less aggressive.
        # Comment these out if you want to use the defaults from nmpc_core.py instead.
        self.ctrl_par.delta_max_deg = 25.0
        self.ctrl_par.delta_rate_max_deg_s = 30.0
        self.ctrl_par.q_delta = 2.0
        self.ctrl_par.r_ddelta = 20.0
        self.ctrl_par.r_ddelta_smooth = 50.0

        self.model_ctrl = ModelParams(
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

        # ---------------- SchedFO2 steering model ----------------
        # Used only in direct torque mode. In external mode, Rory's steering_mpc_node handles torque.
        self.sched_v_kmh = np.array(
            [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 36.0, 40.0, 45.0, 50.0],
            dtype=float,
        )
        self.sched_tau_r = np.array(
            [2.953, 1.386, 0.722, 0.627, 0.437, 0.390, 0.342, 0.295, 0.247, 0.200],
            dtype=float,
        )
        self.sched_kss = np.array(
            [501.9, 314.9, 187.4, 143.2, 105.5, 84.6, 71.2, 63.3, 53.9, 44.1],
            dtype=float,
        )

        self.last_torque_cmd = 0.0
        self.torque_rise_rate = 1.033
        self.torque_fall_rate = 1.833

        # ---------------- Path and controller ----------------
        self.nmpc_path = None
        self.speed_profile = None
        self.ax_profile = None
        self.controller = None

        self.last_path_num_points = None
        self.last_path_length_estimate = None

        csv_file = self.get_parameter("path_csv_file").value
        self.using_csv_path = bool(csv_file)

        if csv_file:
            self.load_path_from_csv(csv_file)
        else:
            self.create_fallback_path()

        if self.use_external_steering_mpc:
            mode = "CASCADE: publishes delta_ref to /path_follower/cmd_vel; Rory publishes /cmd_vel"
        else:
            mode = "DIRECT: NMPC converts delta_ref to torque"

        self.get_logger().info(
            f"NMPC path follower initialized. Mode: {mode}, "
            f"publish_to_cmd_vel={self.publish_to_cmd_vel}, torque_limit={self.torque_limit:.2f}"
        )

    # ============================================================
    # Setup helpers
    # ============================================================

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def create_fallback_path(self):
        x_wp = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
        y_wp = np.array([0, 0, 0, 1, 3, 6, 8, 9, 9, 9], dtype=float)
        self.set_path_from_waypoints(x_wp, y_wp, source_name="fallback path")

    def load_path_from_csv(self, filename):
        try:
            data = np.genfromtxt(filename, delimiter=",", names=True)

            x_wp = np.asarray(data["x"], dtype=float)
            y_wp = np.asarray(data["y"], dtype=float)

            keep = [0]
            for i in range(1, len(x_wp)):
                dx = x_wp[i] - x_wp[keep[-1]]
                dy = y_wp[i] - y_wp[keep[-1]]
                if np.hypot(dx, dy) > 1e-6:
                    keep.append(i)

            x_wp = x_wp[keep]
            y_wp = y_wp[keep]

            if len(x_wp) < 2:
                raise RuntimeError("CSV path contains too few unique points.")

            self.set_path_from_waypoints(
                x_wp,
                y_wp,
                source_name=f"CSV path ({filename})",
            )

        except Exception as err:
            self.get_logger().error(f"Failed to load CSV path '{filename}': {err}")
            self.get_logger().warn("Falling back to internal fallback path.")
            self.create_fallback_path()

    def reset_solver_warm_start(self):
        if self.controller is not None:
            self.controller.reset_warm_start()

    def reset_runtime_state(self):
        self.nmpc_speed_ok = False
        self.consecutive_failures = 0
        self.last_torque_cmd = 0.0
        self.last_valid_delta_ref = 0.0
        self.start_time = None
        self.mode = "IDLE"
        self.reset_solver_warm_start()

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
        self.reset_runtime_state()
        self.publish_path_visualization()

        self.get_logger().info(
            f"Loaded {source_name}: {len(x_wp)} waypoints, length={self.nmpc_path.length:.2f} m"
        )

    # ============================================================
    # Robustness helpers
    # ============================================================

    def stop_safely(self, reason):
        self.get_logger().error(reason)
        self.active = False
        self.nmpc_speed_ok = False
        self.mode = "IDLE"
        self.last_torque_cmd = 0.0
        self.publish_status(False)
        self.reset_solver_warm_start()

        if self.use_external_steering_mpc:
            # Rory's steering_mpc_node reacts to path_following_status=False.
            # Do not publish /cmd_vel from this node in cascade mode.
            self.publish_steering_reference(self.delta)
        else:
            self.publish_cmd(0.0, 0.0)

    def data_is_fresh(self):
        now = self.now_s()
        required_times = [self.last_pose_time, self.last_vehicle_state_time]

        if any(t is None for t in required_times):
            return False

        return all((now - t) < self.max_data_age_s for t in required_times)

    def gyro_is_fresh(self):
        if self.last_gyro_time is None:
            return False
        return (self.now_s() - self.last_gyro_time) < self.max_data_age_s

    def update_yaw_rate_fallback(self):
        if self.gyro_valid and self.gyro_is_fresh():
            return 2.0

        if self.prev_yaw is not None and self.r_from_yaw is not None:
            self.r = float(self.r_from_yaw)
            return 1.0

        self.r = float(self.vx / self.WHEELBASE * np.tan(self.delta))
        return 0.0

    def update_speed_gate(self):
        if not self.nmpc_speed_ok and self.vx >= self.nmpc_enable_speed_mps:
            self.nmpc_speed_ok = True
            self.get_logger().info(f"NMPC speed gate enabled at vx={self.vx:.2f} m/s.")

        if self.nmpc_speed_ok and self.vx <= self.nmpc_disable_speed_mps:
            self.nmpc_speed_ok = False
            self.reset_solver_warm_start()
            self.get_logger().warn(f"NMPC speed gate disabled at vx={self.vx:.2f} m/s.")

    def compute_soft_start_ramp(self):
        if self.start_time is None:
            return 0.0

        elapsed = self.now_s() - self.start_time
        return float(np.clip(elapsed / self.soft_start_duration, 0.0, 1.0))

    def limit_torque_rate(self, u_des, u_prev, dt):
        u_des = float(np.clip(u_des, -1.0, 1.0))
        u_prev = float(np.clip(u_prev, -1.0, 1.0))

        du = u_des - u_prev

        if abs(u_des) > abs(u_prev):
            max_du = self.torque_rise_rate * dt
        else:
            max_du = self.torque_fall_rate * dt

        du_limited = float(np.clip(du, -max_du, max_du))
        return float(np.clip(u_prev + du_limited, -1.0, 1.0))

    def handle_stopping(self, remaining):
        self.pub_stopping_active.publish(make_bool(True))
        self.pub_distance_to_end.publish(make_float64(remaining))

        if self.use_external_steering_mpc:
            # In cascade mode the steering_mpc_node is the only node that should publish /cmd_vel.
            # It will react to path_following_status=False. Be aware that Rory's current implementation
            # may publish full brake when inactive.
            self.active = False
            self.nmpc_speed_ok = False
            self.mode = "IDLE"
            self.last_torque_cmd = 0.0
            self.publish_steering_reference(self.delta)
            self.publish_status(False)
            self.reset_solver_warm_start()
            self.get_logger().info(
                "End of path reached. Disabled path following; external steering_mpc_node handles stop."
            )
            return

        # Direct torque mode: this node handles braking.
        vx = max(0.0, float(self.vx))
        torque_cmd = 0.0

        if vx > 1.0:
            accel_cmd = self.end_brake_cmd_fast
        elif vx > self.vehicle_stopped_speed:
            accel_cmd = self.end_brake_cmd_slow
        else:
            accel_cmd = 0.0
            self.active = False
            self.nmpc_speed_ok = False
            self.mode = "IDLE"
            self.last_torque_cmd = 0.0
            self.publish_status(False)
            self.reset_solver_warm_start()
            self.get_logger().info("Vehicle stopped at end of path.")

        self.publish_cmd(accel_cmd, torque_cmd)

    # ============================================================
    # Callbacks
    # ============================================================

    def pose_callback(self, msg):
        now = self.now_s()
        new_yaw = yaw_from_quaternion(msg.pose.orientation)

        if self.yaw is not None and self.last_pose_time is not None:
            dt = now - self.last_pose_time
            if 1e-3 < dt < 1.0:
                dyaw = wrap_angle(new_yaw - self.yaw)
                self.r_from_yaw = float(dyaw / dt)

        self.prev_yaw = self.yaw
        self.prev_yaw_time = self.last_pose_time

        self.x = float(msg.pose.position.x)
        self.y = float(msg.pose.position.y)
        self.yaw = float(new_yaw)

        self.gnss_valid = True
        self.last_pose_time = now

    def gyro_callback(self, msg):
        self.r = float(msg.vector.z)
        self.gyro_valid = True
        self.last_gyro_time = self.now_s()

    def vehicle_state_callback(self, msg):
        self.vx = max(float(msg.v_ego) / 3.6, 0.0)

        new_delta = (
            float(msg.steering_angle_deg)
            * np.pi / 180.0
            / self.STEERING_RATIO
        )

        now = self.now_s()

        if self.prev_delta is not None and self.prev_delta_time is not None:
            dt = now - self.prev_delta_time
            if 1e-4 < dt < 1.0:
                self.delta_rate = float((new_delta - self.prev_delta) / dt)

        self.prev_delta = new_delta
        self.prev_delta_time = now
        self.delta = float(new_delta)

        self.vehicle_state_valid = True
        self.last_vehicle_state_time = now

    def enable_callback(self, msg):
        if not msg.data:
            self.active = False
            self.nmpc_speed_ok = False
            self.mode = "IDLE"
            self.last_torque_cmd = 0.0
            self.publish_steering_reference(self.delta)
            self.publish_status(False)
            self.reset_solver_warm_start()

            if not self.use_external_steering_mpc:
                self.publish_cmd(0.0, 0.0)

            self.get_logger().info("Path following STOPPED.")
            return

        if not self.active:
            if not self.gnss_valid:
                self.get_logger().warn("Cannot start: no GNSS pose received.")
                return

            if not self.vehicle_state_valid:
                self.get_logger().warn("Cannot start: no vehicle/state received.")
                return

            if self.nmpc_path is None or self.controller is None:
                self.get_logger().warn("Cannot start: no valid path.")
                return

            if not self.gyro_valid:
                self.get_logger().warn(
                    "Starting without fresh /gnss/gyro. Yaw-rate fallback will be used."
                )

            initial_search_length = min(
                self.initial_search_fraction * self.nmpc_path.length,
                self.initial_search_max_length,
            )

            self.last_s = self.find_closest_s(
                self.x,
                self.y,
                last_s=0.0,
                search_radius=initial_search_length,
            )

            self.get_logger().info(
                f"Initial closest s = {self.last_s:.2f} m "
                f"(restricted to first {100.0 * self.initial_search_fraction:.0f}% "
                f"of path, max {self.initial_search_max_length:.1f} m, "
                f"actual search length {initial_search_length:.1f} m)"
            )

            self.active = True
            self.mode = "TRACKING"
            self.nmpc_speed_ok = False
            self.consecutive_failures = 0
            self.last_torque_cmd = 0.0
            self.start_time = self.now_s()

            self.reset_solver_warm_start()
            self.publish_steering_reference(self.delta)
            self.publish_status(True)
            self.pub_stopping_active.publish(make_bool(False))

            self.get_logger().info("Path following STARTED.")
            return

        self.active = False
        self.nmpc_speed_ok = False
        self.mode = "IDLE"
        self.last_torque_cmd = 0.0
        self.publish_steering_reference(self.delta)
        self.publish_status(False)
        self.reset_solver_warm_start()

        if not self.use_external_steering_mpc:
            self.publish_cmd(0.0, 0.0)

        self.get_logger().info("Path following STOPPED by toggle.")

    def path_callback(self, msg):
        if self.active:
            self.get_logger().warn(
                "Received new path while path following is active. Ignoring path update."
            )
            return

        if self.using_csv_path:
            return

        if len(msg.poses) < 2:
            self.get_logger().warn("Received path with fewer than 2 poses. Ignoring.")
            return

        path_num_points = len(msg.poses)

        x_first = msg.poses[0].pose.position.x
        y_first = msg.poses[0].pose.position.y
        x_last = msg.poses[-1].pose.position.x
        y_last = msg.poses[-1].pose.position.y

        path_length_estimate = np.hypot(x_last - x_first, y_last - y_first)

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
            self.get_logger().warn("Received path only had duplicate points. Ignoring.")
            return

        try:
            self.set_path_from_waypoints(x_wp, y_wp, source_name="path topic")
        except Exception as err:
            self.get_logger().warn(f"Failed to build SplinePath from path: {err}")

    # ============================================================
    # Geometry
    # ============================================================

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

    def update_path_progress(self):
        s_candidate = self.find_closest_s(
            self.x,
            self.y,
            last_s=self.last_s,
            search_radius=self.s_search_radius,
        )

        s_jump = s_candidate - self.last_s

        if s_jump > self.max_forward_s_jump:
            self.get_logger().warn(
                f"Rejected large forward s jump: "
                f"last_s={self.last_s:.2f}, candidate={s_candidate:.2f}, "
                f"jump={s_jump:.2f}. Keeping previous s."
            )
            s_current = self.last_s
            self.reset_solver_warm_start()

        elif s_jump < -self.max_backward_s_jump:
            self.get_logger().warn(
                f"Rejected backward s jump: "
                f"last_s={self.last_s:.2f}, candidate={s_candidate:.2f}, "
                f"jump={s_jump:.2f}. Keeping previous s."
            )
            s_current = self.last_s

        else:
            s_current = s_candidate

        self.last_s = s_current
        return s_current

    def compute_errors(self, x_car, y_car, yaw, s):
        x_ref = self.nmpc_path.interp_x(s)
        y_ref = self.nmpc_path.interp_y(s)
        psi_ref = self.nmpc_path.interp_psi(s)

        dx = x_car - x_ref
        dy = y_car - y_ref

        ey = dx * np.sin(psi_ref) - dy * np.cos(psi_ref)
        epsi = wrap_angle(psi_ref - yaw)

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
        pose.header.frame_id = "utm32"
        pose.pose.position.x = float(x_global)
        pose.pose.position.y = float(y_global)
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = float(np.sin(yaw_global / 2.0))
        pose.pose.orientation.w = float(np.cos(yaw_global / 2.0))

        return pose

    # ============================================================
    # Actuator interface for direct torque mode
    # ============================================================

    def steering_ref_to_torque(self, delta_ref_next):
        v_kmh = self.vx * 3.6

        tau = float(np.interp(v_kmh, self.sched_v_kmh, self.sched_tau_r))
        kss_sw_deg = float(np.interp(v_kmh, self.sched_v_kmh, self.sched_kss))

        a = np.exp(-self.ctrl_par.dt / tau)

        kss_front_rad = kss_sw_deg * np.pi / 180.0 / self.STEERING_RATIO
        b = kss_front_rad * (1.0 - a)

        if abs(b) < 1e-8:
            return 0.0

        torque = (delta_ref_next - a * self.delta) / b
        return float(np.clip(torque, -self.torque_limit, self.torque_limit))

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

    # ============================================================
    # Publish helpers
    # ============================================================

    def publish_cmd(self, accel_cmd, torque_cmd):
        msg = Twist()
        msg.linear.x = float(np.clip(accel_cmd, -1.0, 1.0))
        msg.angular.z = float(np.clip(torque_cmd, -1.0, 1.0))

        self.debug_cmd_vel_pub.publish(msg)

        # In cascade mode, Rory's steering_mpc_node must be the only publisher to /cmd_vel.
        if self.publish_to_cmd_vel and not self.use_external_steering_mpc:
            self.cmd_vel_pub.publish(msg)

    def publish_steering_reference(self, delta_ref):
        msg = Twist()
        msg.linear.x = float(self.desired_speed_mps)
        msg.angular.z = float(delta_ref)  # desired front-wheel steering angle [rad]
        self.pub_pf_cmd_vel.publish(msg)

    def publish_status(self, active):
        msg = Bool()
        msg.data = bool(active)
        self.status_pub.publish(msg)

    def publish_path_visualization(self):
        if self.nmpc_path is None:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "utm32"

        for s in np.linspace(0.0, self.nmpc_path.length, 200):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(self.nmpc_path.interp_x(s))
            pose.pose.position.y = float(self.nmpc_path.interp_y(s))

            yaw = self.nmpc_path.interp_psi(s)
            pose.pose.orientation.z = float(np.sin(yaw / 2.0))
            pose.pose.orientation.w = float(np.cos(yaw / 2.0))

            path_msg.poses.append(pose)

        self.path_vis_pub.publish(path_msg)

    def publish_prediction(self, X_opt, s_preview):
        pred_msg = Path()
        pred_msg.header.stamp = self.get_clock().now().to_msg()
        pred_msg.header.frame_id = "utm32"

        for k in range(X_opt.shape[0]):
            ey_k = float(X_opt[k, 0])
            epsi_k = float(X_opt[k, 1])
            s_k = float(s_preview[min(k, len(s_preview) - 1)])

            pose = self.error_state_to_pose(ey_k, epsi_k, s_k)
            pose.header = pred_msg.header
            pred_msg.poses.append(pose)

        self.pred_path_pub.publish(pred_msg)

    def publish_debug_topics(
        self,
        ey,
        epsi,
        delta_ref,
        torque_cmd,
        torque_cmd_raw,
        s_current,
        solve_time,
        yaw_rate_source,
    ):
        cte_rory = ey
        epsi_rory = epsi

        desired_delta_ff = np.arctan(self.veh_ctrl.L * self.nmpc_path.interp_kappa(s_current))
        steering_error = delta_ref - self.delta
        remaining = self.nmpc_path.length - s_current

        self.pub_cte.publish(make_float64(cte_rory))
        self.pub_hdg_err.publish(make_float64(np.rad2deg(epsi_rory)))
        self.pub_desired_delta.publish(make_float64(np.rad2deg(delta_ref)))
        self.pub_actual_delta.publish(make_float64(np.rad2deg(self.delta)))
        self.pub_steering_error.publish(make_float64(np.rad2deg(steering_error)))
        self.pub_delta_ff.publish(make_float64(np.rad2deg(desired_delta_ff)))
        self.pub_torque_cmd.publish(make_float64(torque_cmd))
        self.pub_torque_raw.publish(make_float64(torque_cmd_raw))
        self.pub_progress.publish(make_float64(s_current))
        self.pub_distance_to_end.publish(make_float64(remaining))

        self.pub_lateral_error.publish(make_float64(cte_rory))
        self.pub_heading_error.publish(make_float64(epsi_rory))

        self.pub_solve_time.publish(
            make_float64(1000.0 * solve_time if np.isfinite(solve_time) else np.nan)
        )

        self.pub_speed_gate.publish(make_bool(self.nmpc_speed_ok))
        self.pub_yaw_rate_source.publish(make_float64(yaw_rate_source))
        self.pub_stopping_active.publish(make_bool(self.mode == "STOPPING"))
        self.pub_external_mode.publish(make_bool(self.use_external_steering_mpc))

    # ============================================================
    # Main control loop
    # ============================================================

    def control_loop(self):
        if not self.active:
            return

        if self.x is None or self.y is None or self.yaw is None:
            return

        if self.nmpc_path is None or self.controller is None:
            self.stop_safely("No valid path/controller. Stopping.")
            return

        if not self.data_is_fresh():
            self.stop_safely("Stale input data. Stopping NMPC.")
            return

        yaw_rate_source = self.update_yaw_rate_fallback()

        # Update progress before speed gating so STOPPING can be handled consistently.
        s_current = self.update_path_progress()
        remaining = self.nmpc_path.length - s_current
        self.pub_distance_to_end.publish(make_float64(remaining))

        if self.mode == "TRACKING" and remaining < self.end_slowdown_distance:
            self.get_logger().info(
                f"Approaching end of path. Switching to STOPPING mode. remaining={remaining:.2f} m"
            )
            self.mode = "STOPPING"
            self.nmpc_speed_ok = False
            self.reset_solver_warm_start()

        if self.mode == "STOPPING":
            self.handle_stopping(remaining)
            return

        self.update_speed_gate()

        if not self.nmpc_speed_ok:
            # Below NMPC minimum speed: do not solve NMPC.
            # In cascade mode, keep Rory's steering target close to the measured steering angle.
            self.publish_steering_reference(self.delta)

            if not self.use_external_steering_mpc:
                accel_cmd = self.kp_speed * (self.desired_speed_mps - self.vx)
                accel_cmd = float(np.clip(accel_cmd, 0.0, 1.0))
                self.publish_cmd(accel_cmd, 0.0)

            self.pub_speed_gate.publish(make_bool(False))
            self.pub_yaw_rate_source.publish(make_float64(yaw_rate_source))
            self.pub_stopping_active.publish(make_bool(False))
            self.pub_external_mode.publish(make_bool(self.use_external_steering_mpc))

            self.get_logger().warn(
                f"NMPC held below speed threshold. vx={self.vx:.2f} m/s, "
                f"enable at {self.nmpc_enable_speed_mps:.2f} m/s."
            )
            return

        ey_rory, epsi_rory = self.compute_errors(self.x, self.y, self.yaw, s_current)

        ey_nmpc = -ey_rory
        epsi_nmpc = -epsi_rory

        x0 = np.array(
            [
                ey_nmpc,
                epsi_nmpc,
                self.vy,
                self.r,
                self.delta,
            ],
            dtype=float,
        )

        s_preview, kappa_preview, vx_preview, ax_preview = preview_path_data(
            self.nmpc_path,
            self.speed_profile,
            self.ax_profile,
            x0,
            s_current,
            self.ctrl_par,
            self.model_ctrl,
        )

        # The NMPC prediction currently uses measured speed as a constant preview.
        vx_preview = np.ones_like(vx_preview) * max(self.vx, self.model_ctrl.vx_eps)

        delta_ref = self.delta
        solve_status = "NOT_SOLVED"
        solve_time = np.nan
        iter_count = np.nan

        try:
            X_opt, U_opt, solve_time, iter_count, status, soft_failure = self.controller.solve(
                x0,
                kappa_preview,
                vx_preview,
                ax_preview,
            )

            self.consecutive_failures = 0
            self.publish_prediction(X_opt, s_preview)

            delta_dot = float(U_opt[0, 0])
            delta_ref = self.delta + delta_dot * self.ctrl_par.dt

            delta_ref = float(
                np.clip(
                    delta_ref,
                    -self.controller.delta_max,
                    self.controller.delta_max,
                )
            )

            self.last_valid_delta_ref = delta_ref
            solve_status = status

        except Exception as err:
            self.consecutive_failures += 1

            self.get_logger().warn(
                f"NMPC failed ({self.consecutive_failures}/"
                f"{self.max_consecutive_failures}). Using fallback. Error: {err}"
            )

            self.reset_solver_warm_start()

            if self.consecutive_failures >= self.max_consecutive_failures:
                self.stop_safely("Too many consecutive NMPC failures. Stopping vehicle.")
                return

            delta_ref = self.fallback_delta_ref()
            solve_status = "FALLBACK"

        # Always send NMPC steering reference to Rory's steering_mpc_node in cascade mode.
        if self.use_external_steering_mpc:
            self.publish_steering_reference(delta_ref)
            torque_cmd_raw = 0.0
            torque_cmd = 0.0
        else:
            torque_cmd_raw = self.steering_ref_to_torque(delta_ref)

            ramp = self.compute_soft_start_ramp()
            torque_cmd_raw *= ramp

            torque_cmd = self.limit_torque_rate(
                torque_cmd_raw,
                self.last_torque_cmd,
                self.ctrl_par.dt,
            )
            self.last_torque_cmd = torque_cmd

            accel_cmd = self.kp_speed * (self.desired_speed_mps - self.vx)
            accel_cmd = float(np.clip(accel_cmd, -1.0, 1.0))

            self.publish_cmd(accel_cmd, torque_cmd)

        self.publish_debug_topics(
            ey=ey_rory,
            epsi=epsi_rory,
            delta_ref=delta_ref,
            torque_cmd=torque_cmd,
            torque_cmd_raw=torque_cmd_raw,
            s_current=s_current,
            solve_time=solve_time,
            yaw_rate_source=yaw_rate_source,
        )

        source_name = {
            2.0: "gyro",
            1.0: "yaw_diff",
            0.0: "kinematic",
        }.get(yaw_rate_source, "unknown")

        self.get_logger().info(
            f"mode={self.mode}, external_steering_mpc={self.use_external_steering_mpc}, "
            f"s={s_current:.2f}, remaining={remaining:.2f}, "
            f"ey={ey_rory:.3f}, "
            f"epsi={np.rad2deg(epsi_rory):.2f} deg, "
            f"vx={self.vx:.2f} m/s, "
            f"vy={self.vy:.2f} m/s, "
            f"r={self.r:.3f} rad/s ({source_name}), "
            f"delta={np.rad2deg(self.delta):.2f} deg, "
            f"delta_ref={np.rad2deg(delta_ref):.2f} deg, "
            f"torque_raw={torque_cmd_raw:.3f}, "
            f"torque={torque_cmd:.3f}, "
            f"solve={1000.0 * solve_time:.1f} ms, "
            f"iter={iter_count}, "
            f"status={solve_status}, "
            f"publish_to_cmd_vel={self.publish_to_cmd_vel}"
        )


def main():
    rclpy.init()
    node = NMPCPathFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
