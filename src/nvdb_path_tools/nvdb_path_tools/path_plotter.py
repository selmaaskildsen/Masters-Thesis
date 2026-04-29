import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path

import matplotlib.pyplot as plt


class PathPlotter(Node):

    def __init__(self):
        super().__init__('path_plotter')

        self.subscription = self.create_subscription(
            Path,
            '/reference_path',
            self.path_callback,
            10
        )

        plt.ion()
        self.fig, self.ax = plt.subplots()

        self.get_logger().info("Path plotter started")

    def path_callback(self, msg):
        x = []
        y = []

        for pose in msg.poses:
            x.append(pose.pose.position.x)
            y.append(pose.pose.position.y)

        self.ax.clear()
        self.ax.plot(x, y, 'b-', label='Reference path')
        self.ax.set_title('Reference Path from ROS')
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.axis('equal')
        self.ax.grid(True)
        self.ax.legend()

        plt.draw()
        plt.pause(0.01)


def main(args=None):
    rclpy.init(args=args)

    node = PathPlotter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()