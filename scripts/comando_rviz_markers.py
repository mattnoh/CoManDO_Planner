#!/usr/bin/env python3
"""RViz marker visualization for CoManDO MAVROS runs and bag replays (ROS 1).

The ROS 2 branch got its drone/target visuals from the planner node's
/planner_debug_markers topic; on this branch (and for recorded bags, which
never contain those markers) this standalone node rebuilds them from the
topics that always exist: /mavros/local_position/odom and /target/odom.

Publishes MarkerArray on /comando_viz/markers:
  - target platform: flat red box + trail (frame of /target/odom, 'world')
  - drone: quad glyph (two crossed arms + rotor disks) + trail (frame 'odom')
"""
import math
import rospy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion

TRAIL_LEN = 800
DRONE_AXIS_LEN = 0.34
TARGET_FORWARD_LEN = 0.34
TRAIL_CHUNKS = 7


def quat_mult(a, b):
    return Quaternion(
        w=a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        x=a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        y=a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        z=a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w)


def yaw_quat(yaw):
    return Quaternion(w=math.cos(0.5 * yaw), x=0.0, y=0.0, z=math.sin(0.5 * yaw))


def rotate_vec(q, v):
    # Quaternion-vector rotation for local marker axes. Assumes q is close to unit.
    vx, vy, vz = v
    qw, qx, qy, qz = q.w, q.x, q.y, q.z
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def point_offset(p, v):
    return Point(p.x + v[0], p.y + v[1], p.z + v[2])


class Viz:
    def __init__(self):
        self.drone = None
        self.target = None
        self.drone_trail = []
        self.target_trail = []
        self.target_yaw = 0.0
        self.pub = rospy.Publisher("/comando_viz/markers", MarkerArray, queue_size=2)
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self.on_drone, queue_size=5)
        rospy.Subscriber("/target/odom", Odometry, self.on_target, queue_size=5)
        rospy.Timer(rospy.Duration(0.05), self.tick)

    def on_drone(self, m):
        self.drone = m
        p = m.pose.pose.position
        self.drone_trail.append(Point(p.x, p.y, p.z))
        del self.drone_trail[:-TRAIL_LEN]

    def on_target(self, m):
        self.target = m
        v = m.twist.twist.linear
        if math.hypot(v.x, v.y) > 1e-3:
            self.target_yaw = math.atan2(v.y, v.x)
        p = m.pose.pose.position
        self.target_trail.append(Point(p.x, p.y, p.z))
        del self.target_trail[:-TRAIL_LEN]

    @staticmethod
    def base(frame, ns, mid, mtype):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = rospy.Time(0)  # render at latest transform
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def add_arrow(self, arr, frame, ns, mid, start, end, color, shaft=0.018, head=0.045):
        axis = self.base(frame, ns, mid, Marker.ARROW)
        axis.points = [start, end]
        axis.scale.x = shaft
        axis.scale.y = head
        axis.scale.z = 0.08
        axis.color.r, axis.color.g, axis.color.b, axis.color.a = color
        arr.markers.append(axis)

    def add_faded_trail(self, arr, frame, ns, mid_start, points, rgb, width):
        for marker_id in range(mid_start, mid_start + TRAIL_CHUNKS + 3):
            old = self.base(frame, ns, marker_id, Marker.LINE_STRIP)
            old.action = Marker.DELETE
            arr.markers.append(old)

        if len(points) < 2:
            return

        n = len(points)
        chunk_size = max(2, int(math.ceil(float(n) / TRAIL_CHUNKS)))
        marker_id = mid_start
        for start_idx in range(0, n - 1, chunk_size - 1):
            end_idx = min(n, start_idx + chunk_size)
            if end_idx - start_idx < 2:
                continue

            age = float(end_idx) / float(n)
            trail = self.base(frame, ns, marker_id, Marker.LINE_STRIP)
            trail.points = list(points[start_idx:end_idx])
            trail.scale.x = width * (0.45 + 0.55 * age)
            trail.color.r = rgb[0]
            trail.color.g = rgb[1]
            trail.color.b = rgb[2]
            trail.color.a = 0.10 + 0.55 * age
            arr.markers.append(trail)
            marker_id += 1

    def tick(self, _):
        arr = MarkerArray()

        if self.target is not None:
            frame = self.target.header.frame_id or "world"
            pose = self.target.pose.pose
            box = self.base(frame, "target", 0, Marker.CUBE)
            box.pose.position = pose.position
            box.pose.orientation = yaw_quat(self.target_yaw)
            box.scale.x, box.scale.y, box.scale.z = 0.34, 0.24, 0.035
            box.color.r, box.color.g, box.color.b, box.color.a = 0.86, 0.18, 0.18, 0.9
            arr.markers.append(box)

            deck = self.base(frame, "target", 1, Marker.CUBE)
            deck.pose.position = Point(pose.position.x, pose.position.y, pose.position.z + 0.022)
            deck.pose.orientation = box.pose.orientation
            deck.scale.x, deck.scale.y, deck.scale.z = 0.26, 0.16, 0.012
            deck.color.r, deck.color.g, deck.color.b, deck.color.a = 0.08, 0.08, 0.08, 0.85
            arr.markers.append(deck)

            forward = (
                TARGET_FORWARD_LEN * math.cos(self.target_yaw),
                TARGET_FORWARD_LEN * math.sin(self.target_yaw),
                0.0,
            )
            start = Point(pose.position.x, pose.position.y, pose.position.z + 0.08)
            self.add_arrow(
                arr, frame, "target_forward", 2,
                start, point_offset(start, forward),
                (1.0, 0.75, 0.10, 0.95), shaft=0.014, head=0.04)

            self.add_faded_trail(
                arr, frame, "target_path", 100,
                self.target_trail, (0.95, 0.23, 0.13), 0.016)

        if self.drone is not None:
            frame = self.drone.header.frame_id or "odom"
            pose = self.drone.pose.pose
            q = pose.orientation
            yaw90 = Quaternion(w=math.cos(math.pi/4), x=0, y=0, z=math.sin(math.pi/4))
            for i, arm_q in enumerate((q, quat_mult(q, yaw90))):
                arm = self.base(frame, "drone", 10 + i, Marker.CUBE)
                arm.pose.position = pose.position
                arm.pose.orientation = arm_q
                arm.scale.x, arm.scale.y, arm.scale.z = 0.46, 0.035, 0.026
                arm.color.r, arm.color.g, arm.color.b, arm.color.a = 0.10, 0.36, 0.58, 1.0
                arr.markers.append(arm)

            rotor_offsets = (
                (0.23, 0.0, 0.0),
                (-0.23, 0.0, 0.0),
                (0.0, 0.23, 0.0),
                (0.0, -0.23, 0.0),
            )
            for i, offset in enumerate(rotor_offsets):
                rotor = self.base(frame, "drone_rotors", 30 + i, Marker.CYLINDER)
                rotor.pose.position = point_offset(pose.position, rotate_vec(q, offset))
                rotor.pose.orientation = q
                rotor.scale.x, rotor.scale.y, rotor.scale.z = 0.09, 0.09, 0.012
                rotor.color.r, rotor.color.g, rotor.color.b, rotor.color.a = 0.72, 0.86, 0.96, 0.82
                arr.markers.append(rotor)

            body = self.base(frame, "drone", 12, Marker.CYLINDER)
            body.pose = pose
            body.scale.x, body.scale.y, body.scale.z = 0.13, 0.13, 0.065
            body.color.r, body.color.g, body.color.b, body.color.a = 0.03, 0.18, 0.32, 1.0
            arr.markers.append(body)

            axis_start = Point(pose.position.x, pose.position.y, pose.position.z + 0.06)
            axes = (
                ((DRONE_AXIS_LEN, 0.0, 0.0), (0.95, 0.08, 0.08, 1.0)),  # local X
                ((0.0, DRONE_AXIS_LEN, 0.0), (0.10, 0.72, 0.18, 1.0)),  # local Y
                ((0.0, 0.0, DRONE_AXIS_LEN), (0.15, 0.35, 1.0, 1.0)),  # local Z
            )
            for i, (vec, color) in enumerate(axes):
                self.add_arrow(
                    arr, frame, "drone_axes", 20 + i,
                    axis_start, point_offset(axis_start, rotate_vec(q, vec)),
                    color, shaft=0.012, head=0.04)

            self.add_faded_trail(
                arr, frame, "drone_path", 200,
                self.drone_trail, (0.16, 0.58, 0.94), 0.024)

        if arr.markers:
            self.pub.publish(arr)


if __name__ == "__main__":
    rospy.init_node("comando_rviz_markers")
    Viz()
    rospy.spin()
