# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
from ultralytics import YOLO

DETECT_MASK = True
if DETECT_MASK:
  # det_model = YOLO("yolo26n.pt")
  seg_model = YOLO("yolo11x-seg.pt")

DET_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
TARGET_OBJECT = "bottle"

def yaw_matrix(psi):
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ])

# Camera -> World axis conversion
R_wc = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0]
])


def get_ypr_from_R(R):
    import math
    # Extract angles from rotation matrix
    # For ZYX order in camera frame
    sy = math.sqrt(R[0,0] ** 2 + R[1,0] ** 2)
    singular = sy < 1e-6
    if not singular:
        yaw = math.atan2(R[1,0], R[0,0])
        pitch = math.atan2(-R[2,0], sy)
        roll = math.atan2(R[2,1], R[2,2])
    else:
        yaw = math.atan2(-R[0,1], R[0,0])
        pitch = math.atan2(-R[2,0], sy)
        roll = 0

    yaw_deg = math.degrees(yaw)
    pitch_deg = math.degrees(pitch)
    roll_deg = math.degrees(roll)
    # print(f"Yaw: {yaw_deg:.2f} deg, Pitch: {pitch_deg:.2f} deg, Roll: {roll_deg:.2f} deg")
    return yaw_deg, pitch_deg, roll_deg

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/bottle')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh_file = f'{args.test_scene_dir}/ref_mesh.obj'
  # mesh_file = f'/home/raphael/Projects/github/FoundationPose/demo_data/mustard0/mesh/textured_simple.obj'
  mesh = trimesh.load(mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  # os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  # reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
  rgb_dir = os.path.join(args.test_scene_dir, 'rgb')
  depth_dir = os.path.join(args.test_scene_dir, 'depth')
  masks_dir = os.path.join(args.test_scene_dir, 'masks')
  rgb_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(rgb_dir)) for f in fn if f.endswith(".png") or f.endswith(".jpg")]
  rgb_files.sort()
  depth_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(depth_dir)) for f in fn if f.endswith(".png") or f.endswith(".npy")]
  depth_files.sort()
  masks_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(masks_dir)) for f in fn if f.endswith(".png")]
  masks_files.sort()
  K = np.loadtxt(os.path.join(args.test_scene_dir, 'cam_K.txt')).reshape(3,3)
  assert len(rgb_files) == len(depth_files) == len(masks_files), "The number of rgb, depth, and masks files must be the same"

  object_initial_convention_rotation = None

  started = False
  for i in range(len(rgb_files)):
    logging.info(f'i:{i}')
    color = cv2.imread(rgb_files[i])
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    if depth_files[i].endswith(".npy"):
      depth = np.load(depth_files[i])
    else:
      depth = cv2.imread(depth_files[i], -1)/1e3
    # depth = depth.astype(np.float64)
        
    if i==0 or not started:
      
      
      if DETECT_MASK:
        # Infer the mask using YOLO or RF-DETR or GroundingDino-SAM2 or SAM3
        results = seg_model(color)
        vis = cv2.cvtColor(color, cv2.COLOR_RGB2BGR).copy()  # BGR for cv2 drawing
        np.random.seed(42)
        det_mask_list = []  # store masks for later use (H, W) each
        h_vis, w_vis = vis.shape[0], vis.shape[1]
        for result in results:
          boxes = result.boxes
          masks = result.masks
          if masks is None:
            continue
          mask_data = masks.data.cpu().numpy()  # (num_objects, H, W)
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
                  mode="nearest"
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
        cv2.imshow("Detected objects (masks + class names)", vis)
        # print("Detections shown. Press any key to continue...")
        cv2.waitKey(30)
        # cv2.destroyAllWindows()
        # Use first detection matching CLASS_ID, else first detection, else empty mask
        if det_mask_list and results and results[0].boxes is not None:
          cls_ids = results[0].boxes.cls.cpu().numpy()
          chosen = None
          for idx in range(min(len(cls_ids), len(det_mask_list))):
            if DET_NAMES[int(cls_ids[idx])] == TARGET_OBJECT:
              print(f"Found {TARGET_OBJECT} at index {idx}")
              chosen = idx
              break
            
          if chosen is None:
            print("No target object found, using empty mask")
            mask = np.zeros((color.shape[0], color.shape[1]), dtype=bool)
          else:
            mask = det_mask_list[chosen] > 0
        else:
          mask = np.zeros((color.shape[0], color.shape[1]), dtype=bool)
        
        if chosen is not None:
          started = True
          cv2.destroyAllWindows()
        else:
          print("No target object found, skipping frame")
          continue
      
      else:
        mask = cv2.imread(masks_files[i], cv2.IMREAD_GRAYSCALE)
        mask = mask>0

      tic = time.time()
      if debug==5 and os.path.exists(f'{debug_dir}/ob_in_cam/{i:06d}.txt'):
        pose = np.loadtxt(f'{debug_dir}/ob_in_cam/{i:06d}.txt')
      else:
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)     
      tac = time.time()
      print(f"register time: {tac-tic:.3f} seconds")
    else:
      tic = time.time()
      if debug==5 and os.path.exists(f'{debug_dir}/ob_in_cam/{i:06d}.txt'):
        pose = np.loadtxt(f'{debug_dir}/ob_in_cam/{i:06d}.txt')
      else:
        pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)
      tac = time.time()
      print(f"track time: {tac-tic:.3f} seconds")

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{i:06d}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)


      print("pose:", pose)
      pose_x, pose_y, pose_z = pose[:3,3]
      print("pose_x:", pose_x, "pose_y:", pose_y, "pose_z:", pose_z)
      R_world = R_wc @ pose[:3,:3]
      yaw_before_conv, pitch_before_conv, roll_before_conv = get_ypr_from_R(R_world)
      print("\n--> center_pose:", center_pose)
      
      if i==0 or object_initial_convention_rotation is None:
        if yaw_before_conv > 0 and yaw_before_conv <= 90:
          object_initial_convention_rotation = np.eye(3)
        elif yaw_before_conv > 90 and yaw_before_conv <= 180:
          # rotate -90 degrees around y axis
          object_initial_convention_rotation = yaw_matrix(-np.pi/2)
        elif  yaw_before_conv > -180 and yaw_before_conv <= -90:
          # rotate 180 degrees around y axis
          object_initial_convention_rotation = yaw_matrix(np.pi)
        elif yaw_before_conv > -90 and yaw_before_conv <= 0:
          # rotate 90 degrees around y axis
          object_initial_convention_rotation = yaw_matrix(np.pi/2)
      
      print(object_initial_convention_rotation.shape, R_world.shape)
      R_world = R_world @ object_initial_convention_rotation
      
      yaw_after_conv, pitch_after_conv, roll_after_conv = get_ypr_from_R(R_world)
      print("\n--> yaw_after_conv:", yaw_after_conv, "deg, pitch_after_conv:", pitch_after_conv, "deg, roll_after_conv:", roll_after_conv, "deg")
      
      # R_world_center = R_wc @ center_pose[:3,:3]
      # center_yaw,center_pitch,center_roll = get_ypr_from_R(R_world_center)
      # print("\n--> center_yaw:", center_yaw, "deg, center_pitch:", center_pitch, "deg, center_roll:", center_roll, "deg")
      
      # xx = np.array([1,0,0,1]).astype(float)
      # yy = np.array([0,1,0,1]).astype(float)
      # zz = np.array([0,0,1,1]).astype(float)
      # xx = xx.reshape(4,1)
      # projected_xx = center_pose@xx
      # projected_xx = [np.round(x, 2) for x in projected_xx.reshape(-1)]
      # projected_yy = center_pose@yy
      # projected_yy = [np.round(x, 2) for x in projected_yy.reshape(-1)]
      # projected_zz = center_pose@zz
      # projected_zz = [np.round(x, 2) for x in projected_zz.reshape(-1)]
      # print("projected_xx:", projected_xx)
      # print("projected_yy:", projected_yy)
      # print("projected_zz:", projected_zz)
      # projected = projected.reshape(-1)
      # projected = projected/projected[2]
      # return projected.reshape(-1)[:2].round().astype(int)

      vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)

      ### drawaxis of a fake "0 pose"
      # zero_pose = np.zeros((4,4))
      # zero_pose[3,3] = 1
      # vis = draw_xyz_axis(color, ob_in_cam=zero_pose, scale=0.1, K=K, thickness=5, transparency=0, is_input_rgb=True)

      cv2.putText(vis, f"Yaw_b: {yaw_before_conv:.2f} deg, Pitch_b: {pitch_before_conv:.2f} deg, Roll_b: {roll_before_conv:.2f} deg", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)     
      cv2.putText(vis, f"Yaw_a: {yaw_after_conv:.2f} deg, Pitch_a: {pitch_after_conv:.2f} deg, Roll_a: {roll_after_conv:.2f} deg", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)     

      # cv2.putText(vis, f"Center Yaw: {center_yaw:.2f} deg, Center Pitch: {center_pitch:.2f} deg, Center Roll: {center_roll:.2f} deg", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)     
      # cv2.putText(vis, f"projected_xx: {projected_xx}", (30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)     
      # cv2.putText(vis, f"projected_yy: {projected_yy}", (30,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)     
      # cv2.putText(vis, f"projected_zz: {projected_zz}", (30,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)     
      cv2.putText(vis, f"pose_x: {pose_x:.2f}, pose_y: {pose_y:.2f}, pose_z: {pose_z:.2f}", (30,180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)     

      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(30)
      
    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{i:06d}.png', vis)

