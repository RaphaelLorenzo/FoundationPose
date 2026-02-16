# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
FoundationPose ROS2 node: subscribes to compressed RGB and depth images,
runs object detection (YOLO) and 6-DoF pose estimation (register + track).

Requires: ROS2 (rclpy), sensor_msgs, geometry_msgs, message_filters.
Run from workspace: python run_demo_ros2.py
  (or: ros2 run <your_pkg> run_demo_ros2.py if installed as a package)

Subscriptions:
  - /camera/color/image_raw/compressed (sensor_msgs/CompressedImage)
  - /camera/depth/image_raw/compressed (sensor_msgs/CompressedImage)
  - /camera/color/camera_info (sensor_msgs/CameraInfo) for intrinsics K

Publishes:
  - object_pose (geometry_msgs/PoseStamped)
"""

import os
import time
import math
import threading

import cv2
import imageio
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer

from estimater import *
from ultralytics import YOLO


DET_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog',
    17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
    23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
    43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush',
}


def decode_compressed_color(msg: CompressedImage) -> np.ndarray:
    """Decode CompressedImage to RGB (H, W, 3) uint8."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode color CompressedImage")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_compressed_depth(msg: CompressedImage, scale: float = 0.001) -> np.ndarray:
    """Decode CompressedImage to depth in meters, shape (H, W) float64."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode depth CompressedImage")
    if img.dtype == np.uint16:
        depth = img.astype(np.float64) * scale
    else:
        depth = img.astype(np.float64) / 1000.0
    return depth


class FoundationPoseROS2Node(Node):
    def __init__(self):
        super().__init__("foundation_pose_node")

        self.declare_parameter("mesh_file", "")
        self.declare_parameter("target_object", "bottle")
        self.declare_parameter("est_refine_iter", 5)
        self.declare_parameter("track_refine_iter", 2)
        self.declare_parameter("debug", 1)
        self.declare_parameter("debug_dir", "")
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("color_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw/compressed")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("pose_frame_id", "camera_color_optical_frame")
        self.declare_parameter("slop", 0.1)

        code_dir = os.path.dirname(os.path.realpath(__file__))
        mesh_file = self.get_parameter("mesh_file").value
        if not mesh_file:
            mesh_file = f"{code_dir}/demo_data/bottle/ref_mesh.obj"
        debug_dir = self.get_parameter("debug_dir").value
        if not debug_dir:
            debug_dir = f"{code_dir}/debug"

        self.target_object = self.get_parameter("target_object").value
        self.est_refine_iter = self.get_parameter("est_refine_iter").value
        self.track_refine_iter = self.get_parameter("track_refine_iter").value
        self.debug = self.get_parameter("debug").value
        self.debug_dir = debug_dir
        self.depth_scale = self.get_parameter("depth_scale").value
        self.pose_frame_id = self.get_parameter("pose_frame_id").value
        self.slop = self.get_parameter("slop").value

        os.makedirs(self.debug_dir, exist_ok=True)
        os.makedirs(f"{self.debug_dir}/track_vis", exist_ok=True)
        os.makedirs(f"{self.debug_dir}/ob_in_cam", exist_ok=True)

        self.K = None
        self.est = None
        self.started = False
        self.pose_last = None
        self.to_origin = None
        self.bbox = None
        self.frame_count = 0
        self._lock = threading.Lock()
        self._processing = False

        set_logging_format()
        set_seed(0)

        mesh = trimesh.load(mesh_file)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=glctx,
        )
        self.get_logger().info("FoundationPose estimator initialized")

        self.seg_model = None #YOLO("yolo26x-seg.pt")

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        qos_info = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._camera_info_cb,
            qos_info,
        )

        self._pose_pub = self.create_publisher(
            PoseStamped,
            "object_pose",
            1,
        )

        sub_color = Subscriber(
            self,
            CompressedImage,
            self.get_parameter("color_topic").value,
            qos_profile=qos_sensor,
        )
        sub_depth = Subscriber(
            self,
            CompressedImage,
            self.get_parameter("depth_topic").value,
            qos_profile=qos_sensor,
        )
        self._sync = ApproximateTimeSynchronizer(
            [sub_color, sub_depth],
            queue_size=10,
            slop=self.slop,
        )
        self._sync.registerCallback(self._rgbd_cb)

        self.get_logger().info(
            "Subscribed to %s and %s; waiting for camera_info and RGBD messages"
            % (self.get_parameter("color_topic").value, self.get_parameter("depth_topic").value)
        )

    def _camera_info_cb(self, msg: CameraInfo):
        if self.K is not None:
            return
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.get_logger().info("Received camera intrinsics K")

    def _rgbd_cb(self, color_msg: CompressedImage, depth_msg: CompressedImage):
        if self.K is None:
            return
        if self._lock.acquire(blocking=False):
            if self._processing:
                self._lock.release()
                return
            self._processing = True
            self._lock.release()
        else:
            return

        try:
            color = decode_compressed_color(color_msg)
            depth = decode_compressed_depth(depth_msg, self.depth_scale)
        except ValueError as e:
            self.get_logger().warn(str(e))
            self._lock.acquire()
            self._processing = False
            self._lock.release()
            return

        if color.shape[:2] != depth.shape[:2]:
            depth = cv2.resize(
                depth,
                (color.shape[1], color.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        K = self.K.copy()
        i = self.frame_count
        self.frame_count += 1

        if not self.started:
            results = self.seg_model(color)
            det_mask_list = []
            vis = cv2.cvtColor(color, cv2.COLOR_RGB2BGR).copy()
            h_vis, w_vis = vis.shape[0], vis.shape[1]
            for result in results:
                boxes = result.boxes
                masks = result.masks
                if masks is None:
                    continue
                mask_data = masks.data.cpu().numpy()
                for idx in range(len(boxes)):
                    cls_id = int(boxes.cls[idx].item())
                    name = DET_NAMES.get(cls_id, f"class_{cls_id}")
                    conf = float(boxes.conf[idx].item()) if boxes.conf is not None else 1.0
                    xyxy = boxes.xyxy[idx].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    color_bgr = tuple(int(c) for c in np.random.randint(50, 255, 3))
                    if idx < mask_data.shape[0]:
                        m = mask_data[idx]
                        if m.shape[:2] != (h_vis, w_vis):
                            import torch
                            m = torch.nn.functional.interpolate(
                                torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0),
                                size=(h_vis, w_vis),
                                mode="nearest",
                            ).squeeze().numpy()
                        mask_uint8 = (m > 0.5).astype(np.uint8)
                        det_mask_list.append(mask_uint8)
                        overlay = vis.copy()
                        overlay[mask_uint8 > 0] = (
                            overlay[mask_uint8 > 0] * 0.5 + np.array(color_bgr) * 0.5
                        ).astype(np.uint8)
                        vis = np.where(mask_uint8[:, :, np.newaxis] > 0, overlay, vis).astype(np.uint8)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 2)
                    label = f"{name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color_bgr, -1)
                    cv2.putText(vis, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            chosen = None
            if det_mask_list and results and results[0].boxes is not None:
                cls_ids = results[0].boxes.cls.cpu().numpy()
                for idx in range(min(len(cls_ids), len(det_mask_list))):
                    if DET_NAMES[int(cls_ids[idx])] == self.target_object:
                        self.get_logger().info("Found %s at index %d" % (self.target_object, idx))
                        chosen = idx
                        break

            if chosen is None:
                self.get_logger().warn("No target object '%s' in frame, skipping" % self.target_object)
                self._lock.acquire()
                self._processing = False
                self._lock.release()
                return
            mask = (det_mask_list[chosen] > 0).astype(bool)
            self.started = True

            tic = time.time()
            pose = self.est.register(
                K=K,
                rgb=color,
                depth=depth,
                ob_mask=mask,
                iteration=self.est_refine_iter,
            )
            tac = time.time()
            self.get_logger().info("register time: %.3f s" % (tac - tic))
        else:
            tic = time.time()
            pose = self.est.track_one(
                rgb=color,
                depth=depth,
                K=K,
                iteration=self.track_refine_iter,
            )
            tac = time.time()
            self.get_logger().info("track time: %.3f s" % (tac - tic))

        np.savetxt(f"{self.debug_dir}/ob_in_cam/{i:06d}.txt", pose.reshape(4, 4))

        R = pose[:3, :3]
        t = pose[:3, 3]
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            yaw = math.atan2(R[1, 0], R[0, 0])
            pitch = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[2, 1], R[2, 2])
        else:
            yaw = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            roll = 0
        yaw_deg = math.degrees(yaw)
        pitch_deg = math.degrees(pitch)
        roll_deg = math.degrees(roll)

        if self.debug >= 1:
            center_pose = pose @ np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(
                color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            cv2.putText(
                vis,
                "Yaw: %.2f deg, Pitch: %.2f deg, Roll: %.2f deg" % (yaw_deg, pitch_deg, roll_deg),
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.imshow("foundation_pose", vis[..., ::-1])
            cv2.waitKey(1)
        if self.debug >= 2:
            vis = draw_posed_3d_box(K, img=color, ob_in_cam=pose @ np.linalg.inv(self.to_origin), bbox=self.bbox)
            vis = draw_xyz_axis(
                color,
                ob_in_cam=pose @ np.linalg.inv(self.to_origin),
                scale=0.1,
                K=K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            imageio.imwrite(f"{self.debug_dir}/track_vis/{i:06d}.png", vis)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = color_msg.header.stamp
        pose_msg.header.frame_id = self.pose_frame_id
        pose_msg.pose.position.x = float(t[0])
        pose_msg.pose.position.y = float(t[1])
        pose_msg.pose.position.z = float(t[2])
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(R)
        q = r.as_quat()
        pose_msg.pose.orientation.x = float(q[0])
        pose_msg.pose.orientation.y = float(q[1])
        pose_msg.pose.orientation.z = float(q[2])
        pose_msg.pose.orientation.w = float(q[3])
        self._pose_pub.publish(pose_msg)

        self._lock.acquire()
        self._processing = False
        self._lock.release()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FoundationPoseROS2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
