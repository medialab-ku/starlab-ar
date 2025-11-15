# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from imgui_bundle import imgui
from splatviz_utils.gui_utils import imgui_utils as I
from sandbox_bulletorig import fit_two_stage_from_current_world
from pathlib import Path
from tkinter import filedialog
# sandbox 백엔드: 구체 기반 시뮬 유틸(피팅 없이 GT만 사용)
import sandbox_bullet as sb
import json

# 고정 오브젝트/PLY 경로 (원 코드와 동일)
FIXED_OBJ_PATH = os.path.join("objects", "pokeball", "obj_vhacd.obj")
FIXED_PLY_PATH = os.path.join("objects", "pokeball", "point_cloud.ply")  # (viz에서 필요할 때만)

# -----------------------------
# 좌표 변환 유틸 (Gaussian <-> Bullet)
# -----------------------------
def _to_bullet_pts(scene_quat_xyzw, scene_pos_xyz, pts_g):
    """Gaussian 좌표 pts_g[N,3] -> Bullet(Z-up) 좌표 pts_b[N,3]"""
    P = np.asarray(pts_g, np.float64).reshape(-1, 3)
    q = np.asarray(scene_quat_xyzw, np.float64)  # (x,y,z,w)
    t = np.asarray(scene_pos_xyz,   np.float64)  # (3,)
    Rb = R.from_quat(q)
    Pb = Rb.apply(P) + t
    return Pb

def _to_gaussian_pts(scene_quat_xyzw, scene_pos_xyz, pts_b):
    """Bullet 좌표 pts_b[N,3] -> Gaussian 좌표 pts_g[N,3]"""
    P = np.asarray(pts_b, np.float64).reshape(-1, 3)
    q = np.asarray(scene_quat_xyzw, np.float64)
    t = np.asarray(scene_pos_xyz,   np.float64)
    Rb = R.from_quat(q)
    Pg = Rb.apply(P - t, inverse=True)
    return Pg


# -----------------------------
# Trajectory Widget (GT만)
# -----------------------------
class TrajectoryWidget:
    """
    - GT(구체) 생성
    - GT 저장/로드
    - Gaussian 좌표로 오버레이
    (SPSA/피팅 관련 기능 전부 제거)
    """
    def __init__(self, viz, jsonfile):
        self.viz = viz
        self.name = "Trajectory"

        # UI 토글
        self.show_overlay = True
        self.jsonfile = jsonfile

        # 초기 파라미터(가우시안 좌표)
        self.start0_g = np.array([0.0, 0.0, 0.0], np.float32)
        self.start1_g = np.array([0.0, 0.0, 0.0], np.float32)
        if jsonfile is not None:
            self._on_load_gt_npy(self.jsonfile["gt_traj"])

        # 초기 속도 (Bullet 좌표 기준으로 입력한다고 가정)
        self.v0_0 = np.array([0.0, 0.0, 0.0], np.float32)
        self.v0_1 = np.array([0.0, 0.0, 0.0], np.float32)

        # GT 물성
        self.gt_mass0 = 1.2
        self.gt_mass1 = 2.4
        self.gt_mu    = 0.25
        self.gt_e     = 0.25

        # 시뮬 길이/물리 파라미터
        self.length_sec = 1.5
        self.mu_roll = 0.02
        self.mu_spin = 0.01

        # 최근 I/O 경로/상태
        self.last_gt_path = None
        self.last_save_dir = None

        # GT에서 실제 썼던 Bullet-world 시작좌표(옵션)
        self._last_sim_starts_b = None

        # 디버그 토글
        self.debug_gt = False

        # --- Fit 하이퍼파라미터 (GUI에서 입력받을 값) ---
        self.fit_iters1 = 0
        self.fit_fast_seconds1 = 0.10
        self.fit_stride1 = 1

        self.fit_iters2 = 8
        self.fit_fast_seconds2 = self.length_sec  # None으로 두고 전체 seconds 써도 됨
        self.fit_stride2 = 1
        self.fit_post_tail = 0.5  # loss tail 가중 길이(원본과 동일 의미)
        if self.jsonfile is not None:
            self.fit_iters2 = self.jsonfile["fit_params"]["fit_iters"]
            self.fit_stride2 = self.jsonfile["fit_params"]["fit_stride"]
            self.fit_post_tail = self.jsonfile["fit_params"]["post_tail"]

        self.duration = 0
        self.loss = 0
        self.mu = 0
        self.e = 0
        self.massratio = 0

        self.scene_mesh = str(Path(self.viz.scene_path).with_name("tsdf_fusion_post_vis.obj"))

    # -----------------------------
    # 내부 헬퍼: scene/좌표설정 얻기
    # -----------------------------
    def _get_scene_pose(self):
        # splatviz가 유지하는 scene pose
        scene_q = getattr(self.viz, "scene_origin_quat", [0,0,0,1])  # (x,y,z,w)
        scene_p = getattr(self.viz, "scene_origin_pos",  [0,0,0])
        return scene_q, scene_p

    def _get_sim_params(self):
        dt = float(getattr(self.viz, "traj_dt", 0.01))
        substeps = int(getattr(self.viz, "PHYS_STEPS_PER_FRAME", 8))
        seconds  = float(self.length_sec)
        scene_rotx_deg = float(getattr(self.viz, "rotation", 0.0))
        gravity = (0, 0, -9.8)
        return dt, substeps, seconds, scene_rotx_deg, gravity

    def _resolve_objects_from_load_widget(self):
        """
        무조건 고정 경로의 두 OBJ를 사용한다.
        - 둘 다 동일한 obj_vhacd.obj (두 개 물체)
        - (참고) PLY 경로는 viz에서 필요 시 self.viz에 넘겨 쓰기만 한다.
        """
        obj0 = FIXED_OBJ_PATH
        obj1 = FIXED_OBJ_PATH

        ok = True
        for pth in (obj0, obj1):
            if not isinstance(pth, str) or not os.path.exists(pth):
                ok = False
                break
        if not ok:
            print(f"[two-body-gt] fixed object path missing. expected: {FIXED_OBJ_PATH}")
            return False

        self._last_obj_path0 = obj0
        self._last_obj_path1 = obj1

        # (선택) 다른 위젯/스패너가 참조할 수 있도록 PLY도 알려준다.
        try:
            if hasattr(self.viz, "last_gaussian_ply"):
                setattr(self.viz, "last_gaussian_ply", FIXED_PLY_PATH if os.path.exists(FIXED_PLY_PATH) else None)
        except Exception:
            pass

        return True

    # -----------------------------
    # UI 본체
    # -----------------------------
    @I.scoped_by_object_id
    def __call__(self, expanded: bool):
        expanded, _ = I.collapsing_header("Trajectory", default=True)
        if not expanded:
            return

        changed, val = imgui.checkbox("Show Trajectory Overlay##traj_overlay", bool(self.show_overlay))
        if changed:
            self.show_overlay = bool(val)
        imgui.separator()

        # ----- 시작/속도/GT 파라미터 편집
        self._vec3("start0 (gauss)##traj_s0", "start0_g", (0.0, 0.0, 0.0))
        self._vec3("start1 (gauss)##traj_s1", "start1_g", (0.0, 0.0, 0.0))

        self._vec3("v0[0]##traj_v00", "v0_0", (0.0, 0.0, 0.0))
        self._vec3("v0[1]##traj_v01", "v0_1", (0.0, 0.0, 0.0))

        _, self.gt_mass0 = imgui.input_float("gt mass0##traj_m0", float(self.gt_mass0), 0.05, 0.1)
        _, self.gt_mass1 = imgui.input_float("gt mass1##traj_m1", float(self.gt_mass1), 0.05, 0.1)
        _, self.gt_mu    = imgui.input_float("gt mu##traj_mu",    float(self.gt_mu),    0.02, 0.05)
        _, self.gt_e     = imgui.input_float("gt e##traj_e",      float(self.gt_e),     0.02, 0.05)

        _, self.length_sec = imgui.input_float("seconds##traj_sec", float(self.length_sec), 0.1, 0.25)

        imgui.text(f"Estimated massratio: {self.massratio:.3f}, Mu: {self.mu:.3f}, E: {self.e:.3f}")

        imgui.separator()

        # ----- 파일 I/O 버튼 -----
        if imgui.button("Load GT (.npy)##traj_load", imgui.ImVec2(160, 0)):
            self._on_load_gt_npy(None)
        imgui.same_line()
        if imgui.button("Save GT (.npy)##traj_save", imgui.ImVec2(160, 0)):
            self._on_save_gt_npy()

        # ----- GT 생성 버튼 -----
        debug_changed, debug_val = imgui.checkbox("Debug GT sim (print inputs/first positions)##traj_dbg_gt",
                                                  bool(getattr(self, "debug_gt", False)))
        if debug_changed:
            self.debug_gt = bool(debug_val)

        if imgui.button("Create Trajectory (GT simulate)##traj_create", imgui.ImVec2(330, 0)):
            try:
                self._do_simulate_gt()
            except Exception as e:
                print(f"[two-body-gt] error: {e}")

        imgui.separator()

        # ----- 오버레이 on/off 상태 반영
        self.viz.traj_viz_on = bool(self.show_overlay)

        # ----- Clear
        if imgui.button("Clear (GT)##traj_clear", imgui.ImVec2(330, 0)):
            self._clear_all_trajs_and_caches()

        # ====== 2-Stage Fit 파라미터 입력 ======
        imgui.separator()
        imgui.text("2-Stage SPSA Fitting")

        changed, v = imgui.input_int("fit iters 1##fit_i1", int(self.fit_iters1))
        if changed: self.fit_iters1 = max(0, int(v))
        changed, v = imgui.input_float("fit fast seconds 1##fit_fs1", float(self.fit_fast_seconds1), 0.01, 0.1)
        if changed: self.fit_fast_seconds1 = max(0.0, float(v))
        changed, v = imgui.input_int("fit stride 1##fit_s1", int(self.fit_stride1))
        if changed: self.fit_stride1 = max(1, int(v))

        changed, v = imgui.input_int("fit iters 2##fit_i2", int(self.fit_iters2))
        if changed: self.fit_iters2 = max(0, int(v))
        changed, v = imgui.input_int("fit stride 2##fit_s2", int(self.fit_stride2))
        if changed: self.fit_stride2 = max(1, int(v))

        changed, v = imgui.input_float("fit post tail##fit_tail", float(self.fit_post_tail), 0.1, 0.25)
        if changed: self.fit_post_tail = max(0.0, float(v))

        if imgui.button("Train (2-stage SPSA)##traj_train", imgui.ImVec2(330, 0)):
            try:
                self._on_train_two_stage()
            except Exception as e:
                print(f"[fit] error: {e}")

        imgui.text(f"Loss: {self.loss:.3f}, Time: {self.duration:.3f}")

    # -----------------------------
    # 벡터 입력 helper
    # -----------------------------
    def _vec3(self, label, attr, default):
        v = getattr(self, attr, np.array(default, np.float32))
        ch, vv = imgui.input_float3(label, list(map(float, v)))
        if ch:
            setattr(self, attr, np.array(vv, np.float32))

    # -----------------------------
    # 파일 로드/세이브
    # -----------------------------
    def _on_load_gt_npy(self, file_path):
        if file_path is None:
            path = filedialog.askopenfilename()
        else:
            path = Path(file_path)
        if not path:
            return
        try:
            arr = np.load(path)
            # 기대 shape: [T+1, 2, 3] in Gaussian coords
            if arr.ndim == 3 and arr.shape[1] == 2 and arr.shape[2] == 3:
                self.viz.traj_gt = arr
                self.viz.traj_gt_multi = [arr[:,0,:], arr[:,1,:]]
                self.viz.traj_viz_on = True
                self.last_gt_path = path
                self.start0_g = self.viz.traj_gt_multi[0][0]
                self.start1_g = self.viz.traj_gt_multi[1][0]
                print(f"[traj] loaded GT: {path}")
            else:
                print(f"[traj] npy shape mismatch: {arr.shape}")
        except Exception as e:
            print(f"[traj] load npy error: {e}")

    def _on_save_gt_npy(self):
        if getattr(self.viz, "traj_gt", None) is None:
            print("[traj] no GT to save.")
            return
        save_dir = self.last_save_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "traj_gt.npy")
        try:
            np.save(out, np.asarray(self.viz.traj_gt))
            self.last_save_dir = save_dir
            print(f"[traj] saved GT to {out}")
        except Exception as e:
            print(f"[traj] save error: {e}")

    def _open_file_dialog(self, title, exts=None):
        # 환경마다 구현이 다를 수 있어 간단 대체
        print(title)
        path = input("Enter file path: ").strip()
        if not path:
            return None
        if exts and (os.path.splitext(path)[1].lower() not in exts):
            print("Invalid extension.")
            return None
        if not os.path.exists(path):
            print("File not found.")
            return None
        return path

    def _clear_all_trajs_and_caches(self):
        # viz overlay 비우기
        if hasattr(self.viz, "traj_gt"):
            self.viz.traj_gt = None
        if hasattr(self.viz, "traj_gt_multi"):
            self.viz.traj_gt_multi = []
        # pred/fit 관련 상태는 제거(이 위젯에선 사용 안 함)

        # 내부 캐시/로그 리셋
        self._last_sim_starts_b = None
        print("[traj] cleared GT and caches.")

    # -----------------------------
    # GT 시뮬레이션 (sandbox backend 사용: 구체)
    # -----------------------------
    def _do_simulate_gt(self):
        if not self._resolve_objects_from_load_widget():
            print("[two-body-gt] no object pair selected in Load widget.")
            return

        # 1) scene & sim params
        scene_q, scene_p = self._get_scene_pose()
        dt, substeps, seconds, scene_rotx_deg, gravity = self._get_sim_params()

        # 2) 시작점 (Gaussian -> Bullet)
        g2b = lambda P: _to_bullet_pts(scene_q, scene_p, P)
        s0_b = np.asarray(g2b(self.start0_g), np.float64).reshape(-1, 3)[0].tolist()
        s1_b = np.asarray(g2b(self.start1_g), np.float64).reshape(-1, 3)[0].tolist()
        starts_b = [s0_b, s1_b]

        # 3) sandbox로 GT 시뮬레이션 (현재 월드의 scene_body_id 재사용)
        obj_paths = [FIXED_OBJ_PATH, FIXED_OBJ_PATH]
        scene_mesh = self.scene_mesh
        print("mesh call successful")

        gt_b = sb.bullet_record_traj_multi(
            scene_path=scene_mesh,                      # 파일 경로 불필요 (현재 scene 사용)
            object_paths=obj_paths,
            seconds=seconds, dt=dt, substeps=substeps,
            scene_rotx_deg=scene_rotx_deg,
            gravity=gravity,
            starts=starts_b,
            v0s=[self.v0_0.tolist(), self.v0_1.tolist()],
            mu=float(self.gt_mu), e=float(self.gt_e),
            mu_roll=float(self.mu_roll), mu_spin=float(self.mu_spin),
            masses=[float(self.gt_mass0), float(self.gt_mass1)],
        )

        print("simulation successful")

        # 4) Bullet -> Gaussian 변환 및 viz 반영
        b2g = lambda P: _to_gaussian_pts(scene_q, scene_p, P)
        gt_g = np.asarray(b2g(gt_b.reshape(-1,3)).reshape(gt_b.shape), np.float32)

        self.viz.traj_gt = gt_g
        self.viz.traj_gt_multi = [gt_g[:,0,:], gt_g[:,1,:]]
        self.viz.traj_viz_on = True

        # Pred 시작점으로 동일 재사용할 수 있게 저장(옵션)
        self._last_sim_starts_b = [ starts_b[0], starts_b[1] ]

        print(f"[two-body-gt] simulated {gt_g.shape[0]-1} frames (dt={dt:.5f}, seconds={seconds})")

    def _on_train_two_stage(self):
        # 0) GT가 있어야 학습 가능
        if getattr(self.viz, "traj_gt", None) is None:
            print("[fit] GT trajectory is missing. Create or load GT first.")
            return

        # 1) 공통 scene/params/변환
        scene_q, scene_p = self._get_scene_pose()
        dt, substeps, seconds, scene_rotx_deg, gravity = self._get_sim_params()

        # Gaussian GT → Bullet GT
        g2b = lambda P: _to_bullet_pts(scene_q, scene_p, P)
        gt_g = np.asarray(self.viz.traj_gt, np.float32)  # [T+1,2,3] (Gaussian)
        gt_b = g2b(gt_g.reshape(-1, 3)).reshape(gt_g.shape)  # (Bullet)

        # 시작점(Bullet), 오브젝트 경로
        s0_b = np.asarray(g2b(self.start0_g), np.float64).reshape(-1, 3)[0].tolist()
        s1_b = np.asarray(g2b(self.start1_g), np.float64).reshape(-1, 3)[0].tolist()
        starts_b = [s0_b, s1_b]

        obj_paths = [FIXED_OBJ_PATH, FIXED_OBJ_PATH]

        # 2) 원본 정확도 높은 최적화 함수 호출 (현재 월드 그대로 사용)
        res = sb.simulate(
            scene = self.scene_mesh, objects = obj_paths, seconds = seconds, dt = dt, substeps = substeps,
            rotation = scene_rotx_deg, gravity = gravity, starts = starts_b, gt_traj = gt_b,
            fit_iters1 = self.fit_iters1, fit_iters2 = self.fit_iters2,
            fit_fast_seconds1 = self.fit_fast_seconds1, fit_stride1 = self.fit_stride1, fit_stride2 = self.fit_stride2,
            fit_post_tail = self.fit_post_tail
        )

        v0_est, mu_est, e_est, mu_roll_est, mu_spin_est, m_est = sb.split_params(res[0], 2)
        length = gt_b.shape[0]
        self.duration = res[4]
        self.loss = res[3][1] / length
        self.mu = mu_est
        self.e = e_est
        masses = [round(float(x), 3) for x in m_est.tolist()]
        self.massratio = masses[0] / masses[1]
        run = {"loss": self.loss, "duration":self.duration}

        traj_pred_b = res[1]

        out_path = Path.cwd() / "data" / "gt" / "traj_bullet.npy"
        np.save(out_path, np.asarray(traj_pred_b, dtype=np.float32))
        print(f"[fit] Saved to {out_path}")

        # 5) Bullet → Gaussian 변환 후 오버레이
        b2g = lambda P: _to_gaussian_pts(scene_q, scene_p, P)
        traj_pred_g = np.asarray(b2g(traj_pred_b.reshape(-1, 3)).reshape(traj_pred_b.shape), np.float32)

        # viz에 예측 궤적 올리기 (GT와 구분)
        self.viz.traj_pred = traj_pred_g
        self.viz.traj_pred_multi = [traj_pred_g[:, 0, :], traj_pred_g[:, 1, :]]
        self.viz.traj_viz_on = True

        # --- (추가) best 궤적을 실제 포켓볼 2개로 재생하는 데모 요청 ---
        try:
            # obj_paths 는 위에서 FIXED_OBJ_PATH 두 개로 만들어둔 리스트
            # (예: obj_paths = [FIXED_OBJ_PATH, FIXED_OBJ_PATH])
            if hasattr(self.viz, "start_traj_demo_from_predicted"):
                self.viz.start_traj_demo_from_predicted(
                    traj_pred_b,  # [T+1, 2, 3] (Bullet world)
                    obj_paths,  # 두 공에 대응하는 obj 경로 리스트 (len=2)
                    FIXED_PLY_PATH  # 두 공에 모두 사용할 Gaussian .ply
                )
        except Exception as e:
            print(f"[fit] live demo spawn failed: {e}")

        self._save_runs_as_json(run)

        # # 4) 결과 물성으로 seconds 동안 예측 궤적 시뮬레이션 (현재 월드 재사용)
        # #    - build_bullet_cache(...) → simulate_bullet_once_multi(...) 사용
        # args_base = sb.build_bullet_cache(
        #     scene_obj=None,
        #     object_paths=obj_paths,
        #     scene_rotx_deg=scene_rotx_deg,
        #     start_positions=starts_b,
        #     gravity=gravity,
        #     dt=dt,
        #     substeps=substeps,
        #     scene_body_id=scene_body_id
        # )
        # args_base.update(dict(
        #     seconds=float(seconds),
        #     post_tail=float(self.fit_post_tail),
        #     contact_window=8
        # ))
        #
        # v0_flat = np.array(res["v0"], dtype=np.float32).reshape(-1)
        # traj_pred_b = sb.simulate_bullet_once_multi(
        #     args_base=args_base,
        #     v0_flat=v0_flat,
        #     mu=float(res["mu"]),
        #     e=float(res["e"]),
        #     mu_roll=float(res["mu_roll"]),
        #     mu_spin=float(res["mu_spin"]),
        #     masses=list(map(float, res["masses"])),
        #     seconds_=float(seconds)
        # )

    def _save_runs_as_json(self, data):
        runs_dir = Path(".") / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        name = self.jsonfile["scene_name"]
        out_path = runs_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(data, f)
        print("json saved to {}".format(out_path))