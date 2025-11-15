from imgui_bundle import imgui
import numpy as np
import torch
import sys
import os
import shutil

from trimesh.transformations import quaternion_from_matrix
from scipy.spatial.transform import Rotation as R
import trimesh, pybullet as p, math, pybullet_data

sys.path.append("./gaussian-splatting")
torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2)

from renderer.renderer_wrapper import RendererWrapper
from renderer.gaussian_renderer import GaussianRenderer
from renderer.gan_renderer import GANRenderer

from renderer.attach_renderer import AttachRenderer
from splatviz_utils.gui_utils import imgui_window
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils import gl_utils
from splatviz_utils.gui_utils import text_utils
from splatviz_utils.gui_utils.constants import *
from splatviz_utils.dict_utils import EasyDict
from widgets import (
    edit,
    eval,
    performance,
    load_pkl,
    load_ply,
    camera,
    camera_sequence,
    save,
    latent,
    render,
    training,
    trajectory,
)
from widgets.load_ply import create_physics_object_from_mesh

def local_delta_link(p_b, q_b, p_i, q_i, p_t, q_t):
    # 현재 링크 자세
    q_bt = R.from_quat(q_t) * R.from_quat(q_i).inv()
    p_bt = np.asarray(p_t) - q_bt.apply(p_i)

    # 상대값 (링크 기준)
    q_rel = R.from_quat(q_b).inv() * q_bt
    p_rel = R.from_quat(q_b).inv().apply(p_bt - p_b)
    q_rel_xyzw = q_rel.as_quat()
    q_rel_wxyz = (q_rel_xyzw[3], *q_rel_xyzw[:3])

    return tuple(p_rel), tuple(q_rel_wxyz)

class Splatviz(imgui_window.ImguiWindow):
    def __init__(self, data_path, mode, host, port, gan_path="", scene_path="", objects_path="", rotation=0, caminv=False, jsonfile=None):
        self.code_font_path = "resources/fonts/jetbrainsmono/JetBrainsMono-Regular.ttf"
        self.regular_font_path = "resources/fonts/source_sans_pro/SourceSansPro-Regular.otf"

        super().__init__(
            title="splatviz",
            window_width=1920,
            window_height=1080,
            font=self.regular_font_path,
            code_font=self.code_font_path,
        )

        self.code_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self.regular_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self._imgui_renderer.refresh_font_texture()

        # Internals.
        self._last_error_print = None

        # Determine initial files to load
        initial_files = None
        if scene_path:
            initial_files = []
            # Always include the scene .ply first
            initial_files.append(os.path.abspath(scene_path))
            # Include all .ply files from the objects directory
            if objects_path:
                for fname in os.listdir(objects_path):
                    if fname.endswith(".ply"):
                        initial_files.append(os.path.abspath(os.path.join(objects_path, fname)))
            # Sort object files for consistent order (scene is index 0, objects 1..N)
            # (If a specific order is needed, the user can name files accordingly)
            initial_files[1:] = sorted(initial_files[1:])

        self.widgets = []
        self.rotation = rotation
        self.scene_path = scene_path
        update_all_the_time = True
        self.traj_gt_multi = []
        self.traj_gt = []  # [(x,y,z), ...] in Bullet world
        if mode == "default":
            # Pass initial_files to LoadWidget if provided
            if caminv:
               up_direction = 1
            else:
                up_direction = -1
            self.widgets = [
                load_ply.LoadWidget(self, data_path, initial_files=initial_files),
                camera.CamWidget(self, up_direction=up_direction),
                camera_sequence.CameraSequenceWidget(self),
                performance.PerformanceWidget(self),
                save.CaptureWidget(self),
                render.RenderWidget(self),
                edit.EditWidget(self),
                trajectory.TrajectoryWidget(self, jsonfile),
                eval.EvalWidget(self),
            ]
            renderer = GaussianRenderer()
        elif mode == "attach":
            self.widgets = [
                camera.CamWidget(self),
                camera_sequence.CameraSequenceWidget(self),
                performance.PerformanceWidget(self),
                render.RenderWidget(self),
                edit.EditWidget(self),
                training.TrainingWidget(self),
            ]
            sys.path.append(gan_path)
            renderer = AttachRenderer(host=host, port=port)
            update_all_the_time = True
        elif mode == "gan":
            self.widgets = [
                load_pkl.LoadWidget(self, data_path, file_ending=".pkl"),
                camera.CamWidget(self, fov=12, radius=2.7, up_direction=1),
                camera_sequence.CameraSequenceWidget(self),
                performance.PerformanceWidget(self),
                save.CaptureWidget(self),
                render.RenderWidget(self),
                edit.EditWidget(self),
                eval.EvalWidget(self),
                latent.LatentWidget(self),
            ]
            sys.path.append(gan_path)
            renderer = GANRenderer()
        else:
            raise NotImplementedError(f"Mode '{mode}' not recognized.")

        self.renderer = RendererWrapper(renderer, update_all_the_time)
        self._tex_img = None
        self._tex_obj = None

        # Widget interface.
        self.args = EasyDict()
        self.result = EasyDict()
        self.eval_result = ""

        # After initializing widgets and renderer...
        # Initialize transform list for each loaded scene/object
        num_initial = len(self.widgets[0].plys)
        # Each transform is stored as (quat, trans), quat=[1,0,0,0] (w,x,y,z), trans=[0,0,0] initially
        self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        self.identity_trans = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        self.object_transforms = [(self.identity_quat, self.identity_trans) for _ in range(num_initial)]

        self.dynamic_objects = {}

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame()

        # [click-to-place] 상태 및 파라미터 (튜닝 지점)
        self.awaiting_spawn_click = False
        self.pending_spawn_files = None
        self.spawn_distance_m = 0.5  # ← 기본 소환 거리(미터). 필요시 조절해도 좋음.
        self.throw_impulse_newton = 25000.0  # ← '던지는' 느낌의 힘(뉴턴). 필요시 조절.

        if not hasattr(self, "scene_origin_quat"):
            self.scene_origin_quat = p.getQuaternionFromEuler([math.radians(self.rotation), 0, 0])
        if not hasattr(self, "scene_origin_pos"):
            self.scene_origin_pos = np.zeros(3, dtype=np.float64)

        # ---- trajectory shared state ----

        self.traj_pred = []  # [(x,y,z), ...]
        self.traj_viz_on = False  # toggle
        self.traj_downsample = 1
        self.traj_length = 5.0
        self.traj_recording = False
        self.traj_init_state = {}
        self.traj_learned_params = None
        self.traj_training_done = False

        # --- (추가) fit 결과를 재생하는 데모 상태 ---
        self.demo_traj_b = None  # numpy [T+1, N, 3] (Bullet world)
        self.demo_step = 0  # 현재 재생 중인 스텝 인덱스
        self.demo_bids = []  # 데모에 사용 중인 pybullet body id 들
        self.demo_active = False  # 데모 on/off 플래그
        self.demo_ply_paths = set()

        # 최근 생성된 바디 ID를 trajectory 위젯이 잡아가기 위한 훅
        self.last_spawned_bullet_id = None

        self.bullet_to_path = {}

        self.traj_snapshots = []  # 학습 중 스냅샷 [(N×3 pts, rgba[4]), ...]

        # ✨ 추가: 세션 카운터 & 마지막 소환의 세션
        self.spawn_session_id = 0
        self.last_spawn_session_id = -1



    def close(self):
        for widget in self.widgets:
            widget.close()
        super().close()

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print(f"\n{error}\n")
            self._last_error_print = error

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame()

    def _set_sizes(self):
        self.pane_w = max(self.content_width - self.content_height, 500)
        self.button_w = self.font_size * 5
        self.button_large_w = self.font_size * 10
        self.label_w = round(self.font_size * 5.5) + 100
        self.label_w_large = round(self.font_size * 5.5) + 150

    def set_transform(self, object_idx: int, quat: tuple[float, float, float, float],
                      trans: tuple[float, float, float]) -> None:
        """Update the rotation (quat) and translation for a given object index."""
        if object_idx < 0 or object_idx >= len(self.object_transforms):
            print(f"set_transform: invalid object index {object_idx}")
            return
        # Convert quaternion and translation to torch tensors (on GPU)
        # Expect quat in (w,x,y,z) format
        q = torch.tensor(quat, dtype=torch.float32, device="cuda")
        t = torch.tensor(trans, dtype=torch.float32, device="cuda")
        # Normalize quaternion to be safe
        q = q / torch.norm(q)
        self.object_transforms[object_idx] = (q, t)

    def draw_frame(self):
        self.begin_frame()
        self.args = EasyDict()
        self._set_sizes()

        # Control pane
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(imgui.ImVec2(self.pane_w, self.content_height))
        control_pane_flags = WINDOW_NO_TITLE_BAR | WINDOW_NO_RESIZE | WINDOW_NO_MOVE
        imgui.begin("##control_pane", p_open=True, flags=control_pane_flags)

        # Widgets
        for widget in self.widgets:
            expanded, _visible = imgui_utils.collapsing_header(widget.name, default=widget.name == "Load")
            imgui.indent()
            widget(expanded)
            imgui.unindent()

        # imgui.show_style_editor()

        # **Sync transform list with current number of objects**
        current_files = self.widgets[0].plys
        # Build a map of old file->transform
        prev_map = {}
        if hasattr(self, "object_transforms"):
            prev_files = getattr(self.widgets[0], "prev_plys", None) or current_files
            for j, fname in enumerate(prev_files):
                if j < len(self.object_transforms):
                    prev_map[fname] = self.object_transforms[j]
        # Update prev_plys for next frame
        self.widgets[0].prev_plys = list(current_files)
        # Rebuild object_transforms for current list
        new_transforms = []
        for fname in current_files:
            if fname in prev_map:
                new_transforms.append(prev_map[fname])
            else:
                # New file (added) -> default identity transform
                new_transforms.append((self.identity_quat.clone(),
                                       self.identity_trans.clone()))
        self.object_transforms = new_transforms

        # Pass transforms to renderer via args
        self.args.object_transforms = self.object_transforms

        # Render
        if self.is_skipping_frames():
            pass
        else:
            self.renderer.set_args(**self.args)
            result = self.renderer.result
            if result is not None:
                self.result = result

        # Display
        max_w = self.content_width - self.pane_w
        max_h = self.content_height
        pos = np.array([self.pane_w + max_w / 2, max_h / 2])
        if "image" in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)
        if "error" in self.result:
            self.print_error(self.result.error)
            if "message" not in self.result:
                self.result.message = str(self.result.error)
        if "message" in self.result:
            tex = text_utils.get_texture(
                self.result.message,
                size=self.font_size,
                max_width=max_w,
                max_height=max_h,
                outline=2,
            )
            tex.draw(pos=pos, align=0.5, rint=True, color=1)
        if "eval" in self.result:
            self.eval_result = self.result.eval
        else:
            self.eval_result = None

        # ====== [ANCHOR: after image draw] overlay trajectory ======
        try:
            if getattr(self, "traj_viz_on", False) and self._tex_obj is not None:
                # 이미지 사각형 계산 (우리가 click-to-place에서 쓰던 것과 동일 로직)
                view_w = self.content_width - self.pane_w
                view_h = self.content_height
                tex_w = float(self._tex_obj.width)
                tex_h = float(self._tex_obj.height)
                zoom = min(view_w / tex_w, view_h / tex_h)
                draw_w = tex_w * zoom
                draw_h = tex_h * zoom
                left = pos[0] - draw_w * 0.5
                right = pos[0] + draw_w * 0.5
                top = pos[1] - draw_h * 0.5
                bottom = pos[1] + draw_h * 0.5
                aspect = draw_w / max(1.0, draw_h)

                # 카메라 파라미터
                camw = None
                # 1) 이름으로 우선 탐색
                for w in self.widgets:
                    if getattr(w, "name", "") == "Camera":
                        camw = w
                        break
                # 2) 폴백: 일반 배치(load_ply 다음이 카메라)라면 index=1
                if camw is None and len(self.widgets) > 1:
                    camw = self.widgets[1]
                if camw is not None:
                    fov_y_deg = float(getattr(camw, "fov", 60.0))
                    fov_y = math.radians(fov_y_deg)

                    # 뷰어 월드 기준 기저 (오른손계: right = fwd × up)
                    fwd = getattr(camw, "forward", None)
                    if fwd is None:
                        print("[traj overlay] ERROR: Camera widget has no 'forward'")
                        raise RuntimeError("Camera widget has no 'forward'")
                    fwd = fwd.detach().cpu().numpy().astype(np.float64)
                    fwd /= (np.linalg.norm(fwd) + 1e-9)

                    upv_attr = "up" if hasattr(camw, "up") else ("up_vector" if hasattr(camw, "up_vector") else None)
                    if upv_attr is None:
                        upv = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    else:
                        upv = getattr(camw, upv_attr).detach().cpu().numpy().astype(np.float64)
                    upv /= (np.linalg.norm(upv) + 1e-9)

                    # ★ 오른손계 교정: right = fwd × up,  up' = right × fwd
                    right_w = np.cross(fwd, upv);
                    right_w /= (np.linalg.norm(right_w) + 1e-9)
                    up_w = np.cross(right_w, fwd);
                    up_w /= (np.linalg.norm(up_w) + 1e-9)

                    cam_pos = camw.cam_pos.detach().cpu().numpy().astype(np.float64)

                    def _COL32(r, g, b, a):
                        if hasattr(imgui, "IM_COL32"):
                            return imgui.IM_COL32(int(r), int(g), int(b), int(a))
                        # 폴백: 0~1 float4를 받아들이는 바인딩용
                        return imgui.get_color_u32(
                            imgui.ImVec4(r / 255.0, g / 255.0, b / 255.0, a / 255.0))


                    # 이제 traj 포인트는 이미 'Gaussian(렌더) 좌표'
                    def project_world_point(pt_gauss):
                        Pw = np.array(pt_gauss, dtype=np.float64)

                        # 카메라 좌표 성분
                        v = Pw - cam_pos
                        # ★ 전방 = +Z 해석 (CamWidget.forward 가 씬을 향함)
                        z = np.dot(v, fwd)
                        if z <= 1e-6:
                            return None

                        tanY = math.tan(fov_y * 0.5)
                        sx = left + (0.5 * ((np.dot(v, right_w) / z) / (tanY * aspect) + 1.0)) * draw_w
                        sy = top + (0.5 * (1.0 - (np.dot(v, up_w) / z) / (tanY))) * draw_h
                        return (sx, sy)



                    def draw_polyline(points, rgba, thickness=2.0, step=1):
                        dl = getattr(imgui, "get_foreground_draw_list", imgui.get_background_draw_list)()
                        prev = None
                        for pt in points[::max(1, step)]:
                            sp = project_world_point(pt)
                            if sp is None:
                                prev = None
                                continue
                            if prev is not None:
                                p1 = imgui.ImVec2(prev[0], prev[1])
                                p2 = imgui.ImVec2(sp[0], sp[1])
                                dl.add_line(p1, p2, _COL32(*rgba), thickness)
                            prev = sp

                    # 팔레트 추가(파일 맨 위가 아니라, draw_frame 내부 overlay 블록 안에 둬도 됨)
                    palette = [
                        (0, 0, 255, 255),  # blue
                        (255, 0, 0, 255),  # red
                        (0, 200, 255, 255),  # cyan
                        (255, 165, 0, 255),  # orange
                        (128, 0, 128, 255),  # purple
                        (34, 139, 34, 255),  # green
                    ]

                    # 멀티 GT 먼저 처리
                    gt_multi = getattr(self, "traj_gt_multi", None)
                    if isinstance(gt_multi, (list, tuple)) and len(gt_multi) > 0:
                        for i, seq in enumerate(gt_multi):
                            if len(seq) > 1:
                                draw_polyline(seq, palette[i % len(palette)],
                                              thickness=2.0, step=max(1, self.traj_downsample))
                    else:
                        # 폴백: 단일 GT (None-safety)
                        gt = getattr(self, "traj_gt", [])
                        if isinstance(gt, (list, tuple)) and len(gt) > 1:
                            draw_polyline(gt, (0, 0, 255, 255),
                                          thickness=2.0, step=max(1, self.traj_downsample))
                        elif hasattr(gt, "__len__") and len(gt) > 1:
                            # numpy array 등 list/tuple이 아니더라도 길이가 있으면 허용
                            draw_polyline(gt, (0, 0, 255, 255),
                                          thickness=2.0, step=max(1, self.traj_downsample))

                    # Pred: 멀티 우선
                    pred_multi = getattr(self, "traj_pred_multi", None)
                    if isinstance(pred_multi, (list, tuple)) and len(pred_multi) > 0:
                        for seq in pred_multi:
                            if hasattr(seq, "__len__") and len(seq) > 1:
                                draw_polyline(seq, (0, 255, 0, 255),
                                              thickness=2.0, step=max(1, self.traj_downsample))
                    else:
                        pred = getattr(self, "traj_pred", [])
                        if isinstance(pred, (list, tuple)) and len(pred) > 1:
                            draw_polyline(pred, (255, 255, 255, 255),
                                          thickness=2.0, step=max(1, self.traj_downsample))
                        elif hasattr(pred, "__len__") and len(pred) > 1:
                            draw_polyline(pred, (255, 255, 255, 255),
                                          thickness=2.0, step=max(1, self.traj_downsample))
                    snaps = getattr(self, "traj_snapshots", [])
                    for pts, rgba in snaps:
                        # snapshot_cb는 [0..1] float RGBA를 보냄 → 0..255 정수로 변환
                        if max(rgba) <= 1.0:
                            col = tuple(int(round(c * 255)) for c in rgba)
                        else:
                            col = tuple(int(c) for c in rgba)
                        # 살짝 얇게, 다운샘플은 GT/Pred와 동일하게
                        draw_polyline(pts, col, thickness=1.5, step=max(1, self.traj_downsample))

        except Exception as e:
            print("[trajectory overlay] error:", e)
        # ====== [END overlay trajectory] ======
        # [추가] 클릭-투-플레이스: 뷰포트(렌더 이미지) 클릭 시 스폰
        try:
            if getattr(self, "awaiting_spawn_click",False) and "image" in self.result and self._tex_obj is not None:
                # 렌더 텍스처의 실제 그려진 영역 계산
                view_w = self.content_width - self.pane_w
                view_h = self.content_height
                pos_x, pos_y = pos[0], pos[1]  # draw()에서 사용한 중심좌표
                tex_w = float(self._tex_obj.width)
                tex_h = float(self._tex_obj.height)
                zoom = min(view_w / tex_w, view_h / tex_h)
                draw_w = tex_w * zoom
                draw_h = tex_h * zoom
                left = pos_x - draw_w * 0.5
                right = pos_x + draw_w * 0.5
                top = pos_y - draw_h * 0.5
                bottom = pos_y + draw_h * 0.5

                # 왼쪽 클릭?
                BTN_LEFT = getattr(imgui.MouseButton_, "left", 0)
                if imgui.is_mouse_clicked(BTN_LEFT):
                    mp = imgui.get_mouse_pos()
                    mx, my = mp.x, mp.y  # ImVec2 대응 (언패킹 대신 속성 접근이 안전)
                    # 렌더 이미지 영역 안쪽 클릭만 허용
                    if left <= mx <= right and top <= my <= bottom:
                        # 1) 화면좌표 → NDC(-1~1)
                        u = (mx - left) / draw_w
                        v = (my - top) / draw_h
                        nx = 2.0 * u - 1.0
                        ny = 1.0 - 2.0 * v  # y 반전

                        # 2) 카메라 공간 레이 계산 (수직 FOV 사용)
                        cam_widget = self.widgets[1]  # Load 위젯 다음이 Camera 위젯
                        fov_y_deg = getattr(cam_widget, "fov", 60.0)  # 기본 60도로 폴백
                        fov_y = np.deg2rad(float(fov_y_deg))
                        aspect = draw_w / max(1.0, draw_h)
                        tan_ = np.tan(fov_y * 0.5)

                        # 카메라 기준(-Z 전방) 방향
                        dir_cam = np.array([nx * aspect * tan_, ny * tan_, -1.0], dtype=np.float64)
                        dir_cam /= np.linalg.norm(dir_cam) + 1e-9

                        # 3) 카메라 월드(뷰어) 공간으로 변환
                        # cam_widget.forward/up은 torch 텐서일 가능성 → numpy로
                        fwd = getattr(cam_widget, "forward", None)
                        upv = getattr(cam_widget, "up", None)
                        if fwd is None:
                            raise RuntimeError("Camera widget has no 'forward' vector.")
                        fwd = fwd.detach().cpu().numpy().astype(np.float64)
                        fwd /= np.linalg.norm(fwd) + 1e-9

                        if upv is None:
                            upv = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                        else:
                            upv = upv.detach().cpu().numpy().astype(np.float64)
                        # ★ 동일한 오른손계 정의로 통일
                        right_w = np.cross(fwd, upv);
                        right_w /= np.linalg.norm(right_w) + 1e-9
                        up_w = np.cross(right_w, fwd);
                        up_w /= np.linalg.norm(up_w) + 1e-9

                        # 카메라 공간(-Z 전방)을 월드로 맵핑 (중앙 클릭 → dir_world == fwd)
                        dir_world = (right_w * dir_cam[0] + up_w * dir_cam[1] + fwd * (-dir_cam[2]))
                        dir_world /= np.linalg.norm(dir_world) + 1e-9

                        # 카메라 위치(월드)
                        cam_pos = getattr(cam_widget, "cam_pos", None)
                        if cam_pos is None:
                            raise RuntimeError("Camera widget has no 'cam_pos'.")
                        cam_pos = cam_pos.detach().cpu().numpy().astype(np.float64)

                        # 4) 뷰어→Bullet 월드로 회전 적용 (장면 전역 회전과 일치)
                        if not hasattr(self, "scene_origin_quat"):
                            # 혹시 초기화가 안 된 경우의 안전장치: 프로젝트에서 쓰는 기본 고정값으로 대체
                            # (예: 원래 쓰던 210도 X-회전. 실제 값은 프로젝트와 동일하게 맞추세요)
                            self.scene_origin_quat = p.getQuaternionFromEuler([math.radians(self.rotation), 0, 0])
                        Rg = R.from_quat(self.scene_origin_quat)
                        dir_bullet = Rg.apply(dir_world)
                        cam_bullet = Rg.apply(cam_pos)

                        # 5) 스폰 위치 = 카메라에서 dir로 spawn_distance_m 전방
                        spawn_pos_world = cam_bullet + dir_bullet * float(self.spawn_distance_m)

                        # 6) 준비된 파일로 실제 오브젝트 생성 + 던지기
                        if getattr(self, "pending_spawn_files", None) is None:
                            print("[click-to-place] pending files missing.")
                        else:
                            file_path, obj_path = self.pending_spawn_files
                            bid, com, quat_I = create_physics_object_from_mesh(obj_path, self.scene_origin_quat,
                                                                                   spawn_pos_world)

                            # 렌더러에 등록(초기 pose 지정)
                            self.register_dynamic_object(file_path, bid, com, quat_I,
                                                         init_world_pos=spawn_pos_world,
                                                         init_world_quat=self.scene_origin_quat,
                                                         obj_path=obj_path)

                            # ✨ 추가: 소환 완료 이벤트에 세션을 남김
                            self.last_spawned_bullet_id = bid
                            self.last_spawn_session_id = self.spawn_session_id

                            # impulse(던지기): dir_bullet 방향으로 힘 가하기
                            F = float(getattr(self, "throw_impulse_newton", 10000.0))
                            com_world = spawn_pos_world + Rg.apply(com)
                            p.applyExternalForce(bid, -1, (dir_bullet * F).tolist(), com_world.tolist(),p.WORLD_FRAME)
                            try:
                                self.widgets[0].called_objects += 1
                            except Exception:
                                pass

                        # 상태 리셋
                        self.awaiting_spawn_click = False
                        self.pending_spawn_files = None
        except Exception as e:
            print("[click-to-place] error:", e)
            self.awaiting_spawn_click = False
            self.pending_spawn_files = None

        # End frame.
        self._adjust_font_size()
        imgui.end()
        self.end_frame()

    def register_dynamic_object(self, ply_path, bullet_id, com, quat_I,
                                init_world_pos, init_world_quat, obj_path=None):
        """PyBullet 물체와 해당 .ply 를 연결·초기 변환도 즉시 반영."""
        abs_path = os.path.abspath(ply_path)
        self.dynamic_objects[abs_path] = {
            "id": bullet_id,
            "com": com,
            "quat_I": quat_I,
            "obj_path": obj_path,
            "ply_path": abs_path,
        }
        self.bullet_to_path[bullet_id] = abs_path

        if abs_path not in self.widgets[0].plys:
            self.widgets[0].plys.append(abs_path)

        # <<<< 추가: 새 ply가 들어왔을 때 transform 리스트를 미리 확장 >>>>
        while len(self.object_transforms) < len(self.widgets[0].plys):
            self.object_transforms.append(
                (self.identity_quat.clone(), self.identity_trans.clone())
            )
        # 재구성 단계에서 초기값으로 되돌리지 않도록 prev_plys도 최신화
        self.widgets[0].prev_plys = list(self.widgets[0].plys)
        # <<<< 끝 >>>>

        self.scene_origin_quat = init_world_quat

        rel_pos, rel_quat = local_delta_link(
            [0, 0, 0], init_world_quat, com, quat_I,
            init_world_pos, init_world_quat)

        idx = self.widgets[0].plys.index(abs_path)
        self.set_transform(idx, rel_quat, rel_pos)

        if getattr(self, "awaiting_spawn_click", False):
            self.last_spawned_bullet_id = bullet_id
            self.last_spawn_session_id = self.spawn_session_id

    def remove_dynamic_object_by_bid(self, bid: int, remove_ply: bool = True):
        """Bullet 바디와 Gaussian(=ply 항목)을 함께 제거하고 모든 맵/상태를 정리한다."""
        try:
            path = self.bullet_to_path.get(bid, None)
            # 1) PyBullet 바디 제거
            try:
                p.removeBody(bid)
            except Exception:
                pass
            # 2) 내부 맵/사전 정리
            if path and path in self.dynamic_objects:
                del self.dynamic_objects[path]
            if bid in self.bullet_to_path:
                del self.bullet_to_path[bid]
            # 3) Gaussian(=ply) 제거
            if remove_ply and path and path in self.widgets[0].plys:
                self.widgets[0].plys.remove(path)

                # ---- 추가: 데모용 임시 ply였으면 디스크에서도 삭제 ----
                if hasattr(self, "demo_ply_paths") and path in self.demo_ply_paths:
                    self.demo_ply_paths.discard(path)
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception as e:
                        print("[remove_dynamic_object_by_bid] demo ply remove error:", e)
                # ---------------------------------------------------

            # 4) transform 리스트는 다음 프레임에 plys 기반으로 재구성되므로 별도 조치 불필요
            # 5) 최근 스폰 ID 정리
            if getattr(self, "last_spawned_bullet_id", None) == bid:
                self.last_spawned_bullet_id = None
        except Exception as e:
            print("[remove_dynamic_object_by_bid] error:", e)

    def sync_dynamic_objects(self, scene_origin_pos, scene_origin_quat):
        """매 프레임 PyBullet→Gaussian 좌표 업데이트."""
        # (1) fit 결과 데모가 켜져 있으면 먼저 해당 바디들의 위치를 궤적에 맞게 갱신
        try:
            self._update_demo_bodies()
        except Exception as e:
            print("[sync_dynamic_objects] demo update error:", e)

        # (2) 이후에는 기존 로직대로 모든 dynamic object 를 Gaussian 쪽으로 동기화
        for path, info in self.dynamic_objects.items():
            bid, com, quat_I = info["id"], info["com"], info["quat_I"]
            pos_w, quat_w = p.getBasePositionAndOrientation(bid)
            rel_pos, rel_quat = local_delta_link(
                scene_origin_pos, scene_origin_quat,
                com, quat_I, pos_w, quat_w
            )
            if path in self.widgets[0].plys:
                idx = self.widgets[0].plys.index(path)
                self.set_transform(idx, rel_quat, rel_pos)

    def set_external_camera_pose(self, matrix):
        """외부에서 계산된 카메라 포즈를 현재 뷰어에 적용한다."""
        # Splatviz의 widgets 목록에서 Camera 위젯 찾아서 pose 설정 메서드 호출
        for widget in self.widgets:
            if getattr(widget, "name", "") == "Camera":
                cam_widget = widget
                # 토글이 ON인 경우에만 외부 pose 적용
                if getattr(cam_widget, "auto_pose", True):
                    cam_widget.set_external_camera_pose(matrix)
                # OFF면 적용 건너뜀
                break

    def start_traj_demo_from_predicted(self, traj_pred_b, obj_paths, ply_path):
        """
        trajectory 위젯에서 넘어온 예측 궤적(traj_pred_b)을 기준으로
        포켓볼 오브젝트들을 생성하고, 이후 프레임마다 해당 궤적을 따라가도록 하는 데모를 시작한다.

        traj_pred_b : [T+1, N, 3] (Bullet world 좌표)
        obj_paths   : 길이 N 의 OBJ 경로 리스트
        ply_path    : 모든 공에 사용할 Gaussian .ply 경로 (예: FIXED_PLY_PATH)
        """
        import numpy as np

        traj = np.asarray(traj_pred_b, dtype=np.float64)
        if traj.ndim != 3 or traj.shape[2] != 3:
            print("[demo] invalid traj shape:", traj.shape)
            return

        T, N, _ = traj.shape
        if N == 0:
            print("[demo] empty trajectory (no objects).")
            return
        print("[demo] traj shape:", traj.shape)  # (T+1, 2, 3) 인지 확인
        print("[demo] first step:", traj[0])

        # -------- 0) 베이스 PLY 경로 확인 --------
        base_ply = os.path.abspath(ply_path) if ply_path is not None else None
        if base_ply is None or (not os.path.exists(base_ply)):
            print(f"[demo] invalid ply_path for demo: {ply_path}")
            return

        # -------- 1) 이전 데모 정리 --------
        #  - 기존 데모에서 사용하던 Bullet 바디/ply를 제거
        if getattr(self, "demo_active", False) and self.demo_bids:
            for bid in list(self.demo_bids):
                try:
                    self.remove_dynamic_object_by_bid(bid, remove_ply=True)
                except Exception:
                    pass

        #  - 이전 데모용 ply 파일이 남아있다면 디스크에서 삭제
        if not hasattr(self, "demo_ply_paths"):
            self.demo_ply_paths = set()
        else:
            for old_ply in list(self.demo_ply_paths):
                try:
                    if os.path.exists(old_ply):
                        os.remove(old_ply)
                except Exception as e:
                    print("[demo] failed to remove old demo ply:", old_ply, e)
            self.demo_ply_paths.clear()

        # -------- 2) 새 데모 상태 초기화 --------
        self.demo_traj_b = traj
        self.demo_step = 0
        self.demo_bids = []
        self.demo_active = True

        # scene_origin_quat 이 없으면 기본값 보정
        if not hasattr(self, "scene_origin_quat"):
            self.scene_origin_quat = p.getQuaternionFromEuler(
                [math.radians(self.rotation), 0, 0]
            )

        # -------- 3) 각 물체마다:
        #  - Bullet body 생성
        #  - base_ply 복사본(서로 다른 파일 경로)을 만들어 register_dynamic_object 에 사용
        # --------------------------------------------
        for i in range(N):
            pos0 = traj[0, i, :]  # [3]
            if i < len(obj_paths):
                obj_path_i = obj_paths[i]
            else:
                obj_path_i = obj_paths[-1]

            # 3-1) 데모용 ply 복사본 경로 생성
            dir_name, fname = os.path.split(base_ply)
            name, ext = os.path.splitext(fname)
            demo_ply_i = None

            # "_demo{i}_k" 꼴의 파일명으로 아직 없는 경로를 찾는다
            for k in range(1000):
                candidate = os.path.join(dir_name, f"{name}_demo{i}_{k}{ext}")
                if (candidate not in self.widgets[0].plys) and (not os.path.exists(candidate)):
                    try:
                        shutil.copyfile(base_ply, candidate)
                        demo_ply_i = candidate
                        self.demo_ply_paths.add(candidate)
                    except Exception as e:
                        print("[demo] failed to copy ply for object", i, ":", e)
                        demo_ply_i = base_ply  # 복사가 실패하면 그냥 베이스 경로 재사용(최후 폴백)
                    break

            if demo_ply_i is None:
                # 1000번 시도해도 못 찾으면 그냥 베이스 경로 사용(아주 예외적인 경우)
                demo_ply_i = base_ply

            try:
                # 3-2) Bullet world 에 콜라이더/바디 생성
                bid, com, quat_I = create_physics_object_from_mesh(
                    obj_path_i,
                    self.scene_origin_quat,
                    pos0
                )

                # 3-3) Gaussian renderer 쪽에 dynamic object 등록
                self.register_dynamic_object(
                    demo_ply_i,
                    bid,
                    com,
                    quat_I,
                    init_world_pos=pos0,
                    init_world_quat=self.scene_origin_quat,
                    obj_path=obj_path_i,
                )

                self.demo_bids.append(bid)
                print(f"[demo] spawned body {bid} for object {i} at {pos0}")

            except Exception as e:
                print("[demo] spawn error for object", i, ":", e)

        if not self.demo_bids:
            # 전부 실패하면 데모 off
            self.demo_traj_b = None
            self.demo_active = False
            print("[demo] no Bullet bodies spawned; demo disabled.")

    def _update_demo_bodies(self):
        """
        demo_traj_b 가 설정되어 있으면, 현재 demo_step 에 해당하는 위치로
        데모용 Bullet 바디들을 움직인다.
        (p.resetBasePositionAndOrientation 사용 → sandbox에서 얻은 궤적을 그대로 재생)
        """
        if not getattr(self, "demo_active", False):
            return

        if self.demo_traj_b is None or len(self.demo_bids) == 0:
            self.demo_active = False
            return

        T, N, _ = self.demo_traj_b.shape
        if self.demo_step >= T:
            # 재생이 끝났으면 바디/ply 제거 후 종료
            for bid in list(self.demo_bids):
                try:
                    self.remove_dynamic_object_by_bid(bid, remove_ply=True)
                except Exception:
                    pass
            self.demo_bids = []
            self.demo_traj_b = None
            self.demo_active = False
            print("[demo] finished; bodies removed.")
            return

        step = self.demo_step
        #print(f"[demo] step {self.demo_step}/{T}, pos[0]={self.demo_traj_b[self.demo_step, 0, :]}, pos[1]={self.demo_traj_b[self.demo_step, 1, :]}")

        for i, bid in enumerate(self.demo_bids):
            if i >= N:
                continue
            pos = self.demo_traj_b[step, i, :]  # 현재 스텝에서 물체 i 의 위치 [3]

            try:
                # 현재 바디 orientation 유지 (구체라 크게 중요하진 않음)
                _, quat = p.getBasePositionAndOrientation(bid)
            except Exception:
                quat = self.scene_origin_quat

            try:
                p.resetBasePositionAndOrientation(
                    bid,
                    pos.tolist(),
                    quat
                )
            except Exception as e:
                print(f"[demo] reset error (bid={bid}, step={step}):", e)

        # 다음 프레임에는 다음 스텝 재생
        self.demo_step += 1
