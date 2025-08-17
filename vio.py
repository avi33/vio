import os
import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time
import numpy as np
import cv2

import rosbag
from cv_bridge import CvBridge

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ==========================
# Config
# ==========================
@dataclass
class Config:
    # Camera
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray  # k1,k2,p1,p2[,k3]
    img_size: Tuple[int, int]

    # Feature tracking
    max_features: int = 500
    grid_rows: int = 20
    grid_cols: int = 30
    patch_win: int = 21
    pyr_levels: int = 4
    fb_thresh_px: float = 1.0
    min_eig_thresh: float = 1e-4
    ncc_min: float = 0.85

    # RANSAC
    ransac_thresh_px: float = 1.0
    ransac_conf: float = 0.999
    min_inliers_pose: int = 30

    # IMU noise (typical MEMS; tune to your sensor)
    sigma_g: float = 0.005  # rad/sqrt(s)
    sigma_a: float = 0.05   # m/s^2/sqrt(s)
    sigma_wg: float = 5e-5  # rad/s^1.5
    sigma_wa: float = 5e-4  # m/s^3/2

    # Gravity
    g_mag: float = 9.81

    # Sliding window / MSCKF
    max_clones: int = 12
    pixel_sigma: float = 1.0

    # Keyframe policy
    min_track_keep: float = 0.6
    keyframe_flow_px: float = 30.0

    # Dataset
    dataset_root: str = "./EuRoC/MH_01_easy"

    min_dist = 3
# ==========================
# Camera model utilities
# ==========================
class Pinhole:
    def __init__(self, cfg: Config):
        self.K = np.array([[cfg.fx, 0, cfg.cx], [0, cfg.fy, cfg.cy], [0, 0, 1]], dtype=np.float64)
        self.dist = cfg.dist.astype(np.float64)
        self.size = cfg.img_size
        self.map1, self.map2 = None, None

    def undistort_init(self):
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, self.size, cv2.CV_32FC1)

    def undistort(self, img_gray: np.ndarray) -> np.ndarray:
        if self.map1 is None:
            self.undistort_init()
        return cv2.remap(img_gray, self.map1, self.map2, cv2.INTER_LINEAR)


# ==========================
# Data structures
# ==========================
@dataclass
class ImuSample:
    t: float
    w: np.ndarray  # gyro (rad/s)
    a: np.ndarray  # accel (m/s^2)

@dataclass
class Frame:
    t: float
    img: np.ndarray  # grayscale, undistorted

@dataclass
class Track:
    id: int
    uv: Dict[float, np.ndarray] = field(default_factory=dict)  # t -> (u,v)
    alive: bool = True


# ==========================
# IMU Preintegrator (Forster-style, minimal)
# ==========================
class Preintegrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.dR = np.eye(3)
        self.dv = np.zeros(3)
        self.dp = np.zeros(3)
        self.J_bg = np.zeros((3,3))
        self.J_ba = np.zeros((3,3))
        self.P = np.eye(9) * 1e-6
        self.t0 = None
        self.t1 = None

    @staticmethod
    def _skew(x):
        return np.array([[0,-x[2],x[1]], [x[2],0,-x[0]], [-x[1],x[0],0]])

    def integrate(self, meas: List[ImuSample], bg: np.ndarray, ba: np.ndarray):
        if not meas:
            return
        if self.t0 is None:
            self.t0 = meas[0].t
        for k in range(len(meas)-1):
            m0, m1 = meas[k], meas[k+1]
            dt = m1.t - m0.t
            if dt <= 0: 
                continue
            w = 0.5*(m0.w + m1.w) - bg
            a = 0.5*(m0.a + m1.a) - ba

            # rotation update (first-order)
            theta = w * dt
            angle = np.linalg.norm(theta)
            if angle < 1e-8:
                dRk = np.eye(3)
            else:
                axis = theta / angle
                K = self._skew(axis)
                dRk = np.eye(3) + math.sin(angle)*K + (1-math.cos(angle))*(K@K)

            self.dR = self.dR @ dRk
            self.dv += self.dR @ a * dt
            self.dp += self.dv * dt + 0.5 * (self.dR @ a) * dt * dt

            # TODO: propagate Jacobians (J_bg, J_ba) and covariance P with closed-form
            # For a baseline, keep them zero/small and rely on tuning.
        self.t1 = meas[-1].t


# ==========================
# Feature detection & block optical flow
# ==========================
class BlockFlowTracker:
    def __init__(self, cfg: Config, cam: Pinhole):
        self.cfg = cfg
        self.cam = cam
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.prev_frame: Optional[Frame] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_ids: Optional[np.ndarray] = None

    def _good_corners_grid(self, img: np.ndarray, need: int) -> np.ndarray:
        h, w = img.shape[:2]
        gr, gc = self.cfg.grid_rows, self.cfg.grid_cols
        cells = []
        per_cell = max(1, need // (gr*gc))
        pts_all = []
        for r in range(gr):
            for c in range(gc):
                x0 = int(c*w/gc); x1 = int((c+1)*w/gc)
                y0 = int(r*h/gr); y1 = int((r+1)*h/gr)
                roi = img[y0:y1, x0:x1]
                corners = cv2.goodFeaturesToTrack(
                    roi, maxCorners=per_cell, qualityLevel=0.01, minDistance=5,
                    blockSize=7, useHarrisDetector=False)
                if corners is not None:
                    corners = corners.squeeze(1)
                    corners[:,0] += x0; corners[:,1] += y0
                    pts_all.append(corners)
        if len(pts_all)==0: 
            return np.empty((0,2), np.float32)
        pts = np.vstack(pts_all).astype(np.float32)
        return pts

    # def _spawn_new(self, img: np.ndarray, existing: np.ndarray):
    #     need = max(0, self.cfg.max_features - (0 if existing is None else len(existing)))
    #     if need == 0:
    #         return np.empty((0,2), np.float32), []
    #     pts = self._good_corners_grid(img, need)
    #     # Remove near existing
    #     if existing is not None and len(existing)>0 and len(pts)>0:
    #         dists = cv2.distanceTransform((np.ones(img.shape, np.uint8)*255), cv2.DIST_L2, 3)
    #     return pts, [None]*len(pts)

    # def _spawn_new(self, img: np.ndarray, existing: np.ndarray):
    #     need = max(0, self.cfg.max_features - (0 if existing is None else len(existing)))
    #     if need == 0:
    #         return np.empty((0,2), np.float32), []

    #     pts = self._good_corners_grid(img, need)

    #     # Remove new points that are too close to existing points
    #     if existing is not None and len(existing) > 0 and len(pts) > 0:
    #         mask = np.ones(img.shape, np.uint8) * 255
    #         for e in existing.astype(int):
    #             x, y = e
    #             cv2.circle(mask, (x, y), 5, 0, -1)  # zero-out around existing pts
    #         dists = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    #         keep = []
    #         for p in pts:
    #             x, y = int(p[0]), int(p[1])
    #             if dists[y, x] > 1:  # at least 1 px away
    #                 keep.append(p)
    #         pts = np.array(keep, np.float32)

    #     return pts, [None]*len(pts)

    def _spawn_new(self, img: np.ndarray, existing: np.ndarray):
        need = max(0, self.cfg.max_features - (0 if existing is None else len(existing)))
        if need == 0:
            return np.empty((0, 2), np.float32), []

        h, w = img.shape[:2]
        mask = np.ones((h, w), np.uint8) * 255

        # Draw circles around existing points to block nearby spawning
        if existing is not None and len(existing) > 0:
            for e in existing.astype(int):
                x, y = e
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(mask, (x, y), self.cfg.min_dist, 0, -1)

        # Find good corners only in free regions
        pts = cv2.goodFeaturesToTrack(
            img,
            maxCorners=need,
            qualityLevel=0.01,
            minDistance=self.cfg.min_dist,
            mask=mask
        )

        if pts is None:
            return np.empty((0, 2), np.float32), []

        pts = pts.reshape(-1, 2)
        return pts, [None] * len(pts)

    def track(self, frame: Frame, imu_rot_pred: Optional[np.ndarray]=None):
        img = frame.img
        if self.prev_frame is None:
            # seed
            init_pts = self._good_corners_grid(img, self.cfg.max_features)
            self.prev_pts = init_pts.reshape(-1,1,2)
            ids = []
            for p in init_pts:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = Track(tid, {frame.t: p.copy()})
                ids.append(tid)
            self.prev_ids = np.array(ids, dtype=np.int64)
            self.prev_frame = frame
            return self.prev_ids, self.prev_pts.squeeze(1), self.prev_pts.squeeze(1), np.ones(len(ids), bool)

        lk_win = (self.cfg.patch_win, self.cfg.patch_win)
        term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
        new_pts, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_frame.img, img, self.prev_pts, None,
            winSize=lk_win, maxLevel=self.cfg.pyr_levels-1, criteria=term)
        back_pts, st_b, err_b = cv2.calcOpticalFlowPyrLK(
            img, self.prev_frame.img, new_pts, None,
            winSize=lk_win, maxLevel=self.cfg.pyr_levels-1, criteria=term)
        fb = np.linalg.norm(self.prev_pts - back_pts, axis=2).squeeze(1)
        good = (st.squeeze(1)>0) & (st_b.squeeze(1)>0) & (fb < self.cfg.fb_thresh_px)

        ids = self.prev_ids[good]
        p0 = self.prev_pts[good].squeeze(1)
        p1 = new_pts[good].squeeze(1)

        # RANSAC geom outlier rejection (essential/fundamental)
        if len(p0) >= 8:
            E, inl = cv2.findEssentialMat(p0, p1, self.cam.K, method=cv2.RANSAC,
                                          prob=self.cfg.ransac_conf, threshold=self.cfg.ransac_thresh_px)
            if inl is not None:
                inl = inl.ravel().astype(bool)
                ids, p0, p1 = ids[inl], p0[inl], p1[inl]

        # update tracks
        for tid, uv in zip(ids, p1):
            tr = self.tracks.get(tid)
            if tr is not None:
                tr.uv[frame.t] = uv.copy()
        # spawn new if needed
        active_pts = p1 if len(p1)>0 else np.empty((0,2), np.float32)
        new_pts_spawn, _ = self._spawn_new(img, active_pts)
        if len(new_pts_spawn)>0:
            # No need to track; just insert as new points
            for p in new_pts_spawn:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = Track(tid, {frame.t: p.copy()})
                ids = np.append(ids, tid)
                p0 = np.vstack([p0, p]) if len(p0)>0 else np.array([p], dtype=np.float32)
                p1 = np.vstack([p1, p]) if len(p1)>0 else np.array([p], dtype=np.float32)

        # set prev
        self.prev_frame = frame
        self.prev_pts = p1.reshape(-1,1,2)
        self.prev_ids = ids
        return ids, p0, p1, np.ones(len(ids), bool)

    def get_active_tracks(self, min_obs:int=3) -> List[Track]:
        return [tr for tr in self.tracks.values() if tr.alive and len(tr.uv)>=min_obs]


# ==========================
# MSCKF Estimator (skeleton)
# ==========================
@dataclass
class NominalState:
    R: np.ndarray  # 3x3, world-from-body
    p: np.ndarray  # 3
    v: np.ndarray  # 3
    bg: np.ndarray # 3
    ba: np.ndarray # 3

class MSCKF:
    def __init__(self, cfg: Config, cam: Pinhole):
        self.cfg = cfg
        self.cam = cam
        self.state = NominalState(R=np.eye(3), p=np.zeros(3), v=np.zeros(3), bg=np.zeros(3), ba=np.zeros(3))
        self.P = np.eye(15) * 1e-3
        self.clones: Dict[float, Tuple[np.ndarray,np.ndarray]] = {}  # t -> (R,p)

    def propagate_imu(self, imu: ImuSample, dt: float):
        # nominal (simple Euler)
        w = imu.w - self.state.bg
        a = imu.a - self.state.ba
        self.state.R = self.state.R @ self._Exp(w*dt)
        self.state.v = self.state.v + (self.state.R @ a + np.array([0,0,-self.cfg.g_mag]))*dt
        self.state.p = self.state.p + self.state.v*dt + 0.5*(self.state.R @ a + np.array([0,0,-self.cfg.g_mag]))*dt*dt
        # TODO: error-state P propagation (F,G)

    @staticmethod
    def _Exp(theta: np.ndarray) -> np.ndarray:
        a = np.linalg.norm(theta)
        if a < 1e-8:
            return np.eye(3)
        k = theta / a
        K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
        return np.eye(3) + math.sin(a)*K + (1-math.cos(a))*(K@K)

    def clone_pose(self, t: float):
        self.clones[t] = (self.state.R.copy(), self.state.p.copy())
        if len(self.clones) > self.cfg.max_clones:
            # drop the oldest
            t_old = sorted(self.clones.keys())[0]
            self.clones.pop(t_old)

    def update_msckf(self, tracks: List[Track]):
        # TODO: implement nullspace projection residuals and EKF update
        # Placeholder: no-op
        pass


# ==========================
# EuRoC dataset reader (simple)
# ==========================
class EuRoC:
    def __init__(self, root: str):
        self.root = root
        self.cam_dir = os.path.join(root, "mav0", "cam0")
        self.imu_dir = os.path.join(root, "mav0", "imu0")
        self.cam_csv = os.path.join(self.cam_dir, "data.csv")
        self.imu_csv = os.path.join(self.imu_dir, "data.csv")
        # self.cam_list = self._read_csv(self.cam_csv)
        # self.imu_list = self._read_csv(self.imu_csv)
        self.cam_list, self.imu_list = self._read_rosbag(root + '/V2_02_medium.bag', None)
        # convert ns to seconds
        # self.cam_list = [(ts*1e-9, os.path.join(self.cam_dir, "data", fn)) for ts, fn in self.cam_list]
        # self.imu_list = [(ts*1e-9, w, a) for ts,(w,a) in self.imu_list]

    @staticmethod
    def _read_csv(path):
        out = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                ts = int(parts[0])
                if len(parts)==2:  # image
                    out.append((ts, parts[1]))
                else:  # imu: ts, wx,wy,wz, ax,ay,az
                    w = np.array(list(map(float, parts[1:4])), dtype=np.float64)
                    a = np.array(list(map(float, parts[4:7])), dtype=np.float64)
                    out.append((ts, (w,a)))
        return out
    
    @staticmethod
    def _read_rosbag(bag_path: str, export_dir: str):
        """Read ROS bag and store images as PNG to match CSV interface."""
        IMAGE_TOPIC = "/cam0/image_raw"
        IMU_TOPIC   = "/imu0"        

        bag = rosbag.Bag(bag_path)
        bridge = CvBridge()
        cam_list = []
        imu_list = []        
        for topic, msg, _ in bag.read_messages(topics=[IMAGE_TOPIC, IMU_TOPIC]):
            if topic == IMAGE_TOPIC:
                ts = msg.header.stamp.to_sec()
                img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                # fname = f"{int(ts*1e9)}.png"
                # fpath = os.path.join(export_dir, fname)
                # cv2.imwrite(fpath, img)
                cam_list.append((ts, img))
                # data_list.append((ts, img))
            elif topic == IMU_TOPIC:
                ts = msg.header.stamp.to_sec()
                w = np.array([msg.angular_velocity.x,
                              msg.angular_velocity.y,
                              msg.angular_velocity.z], dtype=np.float64)
                a = np.array([msg.linear_acceleration.x,
                              msg.linear_acceleration.y,
                              msg.linear_acceleration.z], dtype=np.float64)
                imu_list.append((ts, w, a))
                # data_list.append((ts, img))

        bag.close()
        return cam_list, imu_list


# ==========================
# Orchestrator
# ==========================
class VIO:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cam = Pinhole(cfg)
        self.cam.undistort_init()
        self.tracker = BlockFlowTracker(cfg, self.cam)
        self.est = MSCKF(cfg, self.cam)
        self.preint = Preintegrator(cfg)

    def run_euroc(self, dataset_root: str):
        ds = EuRoC(dataset_root)
        imu_idx = 0
        fps_camera = (1/np.diff(([t for t, _ in ds.cam_list])).mean()).round()        
        keyframes: List[Frame] = []
        tracks: List[np.ndarray] = []
        h, w = ds.cam_list[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', etc.
        f_out = cv2.VideoWriter('result.mp4', fourcc, fps_camera, (w, h))
        # Iterate over images; integrate all IMU samples up to each image
        frame_id = 0        
        for t_img, img in tqdm(ds.cam_list, desc="Frames"):
            start_loop = time.time()
            # integrate imu up to this frame
            imu_between: List[ImuSample] = []
            while imu_idx < len(ds.imu_list) and ds.imu_list[imu_idx][0] <= t_img:
                t, w, a = ds.imu_list[imu_idx]
                imu_between.append(ImuSample(t=t, w=w, a=a))
                # propagate estimator nominal at high-rate too (optional)
                if imu_idx>0:
                    dt = ds.imu_list[imu_idx][0] - ds.imu_list[imu_idx-1][0]
                    self.est.propagate_imu(ImuSample(t, w, a), dt)
                imu_idx += 1

            # preintegrate between last frame and this frame (for later use)
            self.preint.reset()
            self.preint.integrate(imu_between, self.est.state.bg, self.est.state.ba)

            # read + undistort image
            # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_u = self.cam.undistort(img)
            frame = Frame(t=t_img, img=img_u)

            # track features
            ids, p0, p1, mask = self.tracker.track(frame)

            # keyframe policy (simple)
            median_flow = 0.0 if len(p0)==0 else float(np.median(np.linalg.norm(p1 - p0, axis=1)))
            keep_ratio = 0 if self.tracker.prev_ids is None else len(ids) / max(1,len(self.tracker.prev_ids))
            make_keyframe = (median_flow > self.cfg.keyframe_flow_px) or (keep_ratio < self.cfg.min_track_keep)
            if make_keyframe:
                keyframes.append(frame)
                new_pts, new_meta = self.tracker._spawn_new(img_u, p1)
                tracks.extend(new_pts)

            # clone pose and update (placeholder)
            self.est.clone_pose(t_img)
            active_tracks = self.tracker.get_active_tracks(min_obs=3)
            self.est.update_msckf(active_tracks)
            proc_time = time.time() - start_loop
            proc_fps = 1.0 / proc_time if proc_time > 0 else 0.0            
            # (Optional) visualize
            vis = cv2.cvtColor(img_u, cv2.COLOR_GRAY2BGR)
            for uv in p1.astype(int):
                cv2.circle(vis, tuple(uv), 2, (0,255,0), -1)

            cv2.putText(vis,
                            f"t={t_img-ds.cam_list[0][0]:.1f}s flow={median_flow:.1f}px tracks={len(ids)} FPS={proc_fps:.1f}",
                            (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA)

            
            
            cv2.imshow('tracks', vis)
            f_out.write(vis)            
            key = cv2.waitKey(1)
            if key == 27:
                break
            elapsed = time.time() - start_loop
            sleep_time = max(0, 1/fps_camera - elapsed)
            time.sleep(sleep_time)
            frame_id += 1
            if frame_id > 500:
                break

        cv2.destroyAllWindows()
        f_out.release()

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    # Example intrinsics (EuRoC cam0)
    cfg = Config(
        fx=458.654, fy=457.296, cx=367.215, cy=248.375,
        dist=np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0], dtype=np.float64),
        img_size=(752, 480),
        dataset_root=os.environ.get('EUROC_ROOT', '../../datasets/vio/')
    )

    vio = VIO(cfg)
    print("Starting VIO on:", cfg.dataset_root)
    if not os.path.exists(cfg.dataset_root):
        print("Dataset not found. Set EUROC_ROOT env or edit cfg.dataset_root.")
        exit(1)

    vio.run_euroc(cfg.dataset_root)
