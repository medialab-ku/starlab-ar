# run_main.py 수정안: scene_path + object_path 두 디렉토리에서 ply를 로드

import click
from pathlib import Path
from splatviz import Splatviz
import time
import threading
import trimesh, numpy as np, pybullet as p, math, os, pybullet_data
#from sdgs.sdgs_predictor import SixDGSPredictor
from pybullet_utils import bullet_client
from trimesh.transformations import quaternion_from_matrix
from scipy.spatial.transform import Rotation as R
import cv2
import json

def convert_ply_to_obj(ply_path: str, out_obj_path: str,
                       grid_fine: int = 256, grid_coarse: int = 128,
                       lvl_fine: float = 0.18, lvl_coarse: float = 0.05):
    """
    point‑cloud *.ply* → watertight *.obj*  (gaussian splatting friendly)
    Params are 그대로 기본값 사용.
    """
    import open3d as o3d, numpy as np, plyfile, skimage.measure as mc, os

    ply   = plyfile.PlyData.read(ply_path)
    xyz   = np.stack([ply['vertex'][ax] for ax in ('x','y','z')], 1)
    alpha = np.where(np.isfinite(ply['vertex']['opacity']) &
                     (ply['vertex']['opacity']>0),
                     ply['vertex']['opacity'], 0) * 0.5
    sigma_log = np.stack([ply['vertex'][f'scale_{i}'] for i in range(3)], 1)
    sigma_lin = 2 ** sigma_log

    RES, PAD = 256, 0.05
    bmin, bmax = xyz.min(0)-PAD, xyz.max(0)+PAD

    def voxelize(grid_res, min_sigma_scale):
        vox = np.zeros((grid_res,)*3, np.float32)
        vsize = (bmax-bmin)/(grid_res-1)
        minσ  = vsize.min()*min_sigma_scale
        for p,a,s in zip(xyz,alpha,sigma_lin):
            if a==0: continue
            σ = np.maximum(s,minσ)
            lo = np.floor(((p-σ)-bmin)/vsize).astype(int)
            hi = np.ceil(((p+σ)-bmin)/vsize).astype(int)+1
            xs,ys,zs=[np.clip(np.arange(l,h),0,grid_res-1) for l,h in zip(lo,hi)]
            gx,gy,gz=np.meshgrid(xs,ys,zs,indexing='ij')
            coords=np.stack([gx,gy,gz],-1)*vsize+bmin
            d2=((coords-p)**2/(σ**2+1e-9)).sum(-1)
            vox[np.ix_(xs,ys,zs)] += (d2<1).astype(np.float32)*np.maximum(a,0.05)
        vox/=vox.max()
        return vox

    # coarse + fine
    vox_c = voxelize(grid_coarse, 2.0)
    verts_c,faces_c,_,_ = mc.marching_cubes(vox_c, level=lvl_coarse)
    vox_f = voxelize(grid_fine, 1.2)
    verts_f,faces_f,_,_ = mc.marching_cubes(vox_f, level=lvl_fine)

    scale_c = (bmax-bmin)/(grid_coarse-1); scale_f = (bmax-bmin)/(grid_fine-1)
    mesh_c = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts_c*scale_c + bmin),
                o3d.utility.Vector3iVector(faces_c))
    mesh_f = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts_f*scale_f + bmin),
                o3d.utility.Vector3iVector(faces_f))
    mesh=(mesh_c+mesh_f).remove_duplicated_vertices().filter_smooth_taubin(5)
    os.makedirs(os.path.dirname(out_obj_path), exist_ok=True)
    o3d.io.write_triangle_mesh(out_obj_path, mesh)

def setup_static_scene(scene_ply: Path, world_quat):
    """Load the static scene mesh sitting next to *scene_ply*.

    Expected sibling file name: ``tsdf_fusion_post_vis.obj``.  Returns *scene_uid*.
    """
    scene_mesh = scene_ply.with_name("tsdf_fusion_post_vis.obj")
    assert scene_mesh.is_file(), f"Scene mesh '{scene_mesh}' not found"

    sc_col = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=str(scene_mesh),
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
    )
    sc_vis = p.createVisualShape(p.GEOM_MESH, fileName=str(scene_mesh))

    uid = p.createMultiBody(
        baseMass=0,  # static
        baseCollisionShapeIndex=sc_col,
        baseVisualShapeIndex=sc_vis,
        basePosition=[0, 0, 0],
        baseOrientation=world_quat,
    )
    if uid < 0:
        raise RuntimeError(f"Failed to load scene mesh '{scene_mesh}'")
    return uid

def vhacd_if_needed(src_obj: Path, dst_obj: Path):
    """Run VHACD on *src_obj* → *dst_obj* unless *dst_obj* already exists."""
    if dst_obj.exists():
        return
    ok = p.vhacd(
        str(src_obj),
        str(dst_obj),
        str(dst_obj.with_suffix(".log")),
        concavity=0.002,
        resolution=1_000_000,
    )

def inertia_quat(mesh: trimesh.Trimesh):
    """Principal‑axis quaternion of *mesh*'s inertia tensor."""
    I = mesh.moment_inertia
    _, eigvec = np.linalg.eigh(I)
    # Build a 3×3 rotation matrix from eigen‑vectors
    R_I = np.ascontiguousarray(eigvec)
    return quaternion_from_matrix(
        np.vstack((np.hstack((R_I, [[0], [0], [0]])), [0, 0, 0, 1]))[:3, :3]
    ).tolist()

def estimate_mass_from_mesh(mesh: trimesh.Trimesh,
                            density: float = 500.0,   # kg/m^3
                            unit_scale: float = 1.0,  # e.g., cm→m 이면 0.01
                            min_mass: float = 0.1,
                            max_mass: float = 500.0):
    """메쉬 체적과 밀도로 질량(kg) 추정. 불안정 시 convex hull로 폴백."""
    # 체적(m^3)로 변환: (unit_scale^3) 적용
    vol = float(mesh.volume) * (unit_scale ** 3)
    if (not np.isfinite(vol)) or vol <= 0:
        vol = float(mesh.convex_hull.volume) * (unit_scale ** 3)

    mass = density * vol
    # 안전 클램프
    if not np.isfinite(mass) or mass <= 0:
        mass = min_mass
    mass = float(np.clip(mass, min_mass, max_mass))
    return mass

def load_dynamic_objects(objects_dir: Path, splatviz: Splatviz, world_quat):
    """Discover every ``point_cloud_*.ply`` in *objects_dir* and register them."""

    for ply_path in sorted(objects_dir.glob("point_cloud_*.ply")):
        idx = ply_path.stem.split("_")[-1]  # handles arbitrary index names
        obj_mesh = ply_path.with_suffix(".obj")
        if not obj_mesh.is_file():  # <─ 변경 블록 시작
            print(f"[INFO] OBJ not found for {ply_path.name} → gaustomesh meshing")
            convert_ply_to_obj(str(ply_path), str(obj_mesh))
        vhacd_mesh = objects_dir / f"obj_vhacd_{idx}.obj"

        if not obj_mesh.is_file():
            print(f"[WARN] Missing OBJ for {ply_path.name}, skipping")
            continue

        vhacd_if_needed(obj_mesh, vhacd_mesh)

        # ------------------------------------------------------------------
        # Physics
        # ------------------------------------------------------------------
        mesh = trimesh.load(vhacd_mesh, force="mesh")
        mass = estimate_mass_from_mesh(
            mesh, density=400, unit_scale=1,
            min_mass=0.1, max_mass=500
        )
        com = mesh.center_mass  # [x, y, z]  in local‑mesh frame
        quat_I = inertia_quat(mesh)

        col_id = p.createCollisionShape(p.GEOM_MESH, fileName=str(vhacd_mesh))
        vis_id = p.createVisualShape(p.GEOM_MESH, fileName=str(vhacd_mesh))

        uid = p.createMultiBody(
            baseMass=10,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[0, 0, 0],
            baseOrientation=world_quat,
            baseInertialFramePosition=com.tolist(),
            baseInertialFrameOrientation=quat_I,
        )
        print(f"mass for obj {uid}: {mass}")
        p.changeDynamics(
            uid,
            -1,
            lateralFriction=1.5,  # ↑ 마찰 강화 (1~2 권장)
            rollingFriction=0.05,  # ↑ 굴림 마찰
            spinningFriction=0.05,  # ↑ 스핀 마찰
            linearDamping=0.01,  # ↓ 덜 답답하게
            angularDamping=0.01,  # ↓
            frictionAnchor=1,
            restitution=1
        )

        # ------------------------------------------------------------------
        # Splatviz registration
        # ------------------------------------------------------------------
        splatviz.register_dynamic_object(
            str(ply_path.resolve()),
            uid,
            com,
            quat_I,
            init_world_pos=[0, 0, 0],
            init_world_quat=world_quat,
        )
        print(f"[INFO] Registered dynamic object {ply_path.name} → uid {uid}")


@click.command()
@click.option("--data_path", default="./resources/sample_scenes", help="root path for .ply files", metavar="PATH")
@click.option("--scene_path", default="", help="Path to a single scene .ply file", type=click.Path())
@click.option("--objects_path", default="", help="Path to a directory of object .ply files", type=click.Path())
@click.option("--rotation", default=0, help="Degree of rotation of scene and object files")
@click.option("--checkpoint_path", default="", help="Folder of id_module.th", type=str)
@click.option("--mode", default="default", type=str, help="viewer mode")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=7860)
@click.option("--gan_path", default=None)
@click.option("--video", default="", help="Video file path (if empty, use webcam 0)", type=str)
@click.option("--caminv", default=False)
@click.option("--jsonfile", default=None)
def main(data_path, scene_path, objects_path, checkpoint_path, mode, host, port, gan_path, jsonfile, video=0, rotation=0, caminv=False):
    cfg = None
    if jsonfile is not None:
        with open(jsonfile, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        scene_path = cfg["scene_path"]  # str
        rotation = cfg["rotation"]  # int
        caminv = cfg["caminv"]  # bool
    print(scene_path)
    print(rotation)
    print(caminv)

    #physics
    p.connect(p.DIRECT)  # 또는 p.DIRECT
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # plane.urdf 등
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    world_quat = p.getQuaternionFromEuler([math.radians(rotation), 0, 0])

    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / 240.0,  # 물리 적분 간격 (작게)
        numSubSteps=2,  # 내부 서브스텝
        numSolverIterations=150,  # 이미 있던 값 유지
        useSplitImpulse=1,  # 관통 안정화
        splitImpulsePenetrationThreshold=-0.02,
        restitutionVelocityThreshold=0.05,  # 미세 충돌 반발 억제
        allowedCcdPenetration=0.002,  # CCD 허용 관통
        enableConeFriction=1,
        contactBreakingThreshold=0.02,
    )

    if scene_path:
        scene_ply = Path(scene_path)
        setup_static_scene(scene_ply, world_quat)

    # If scene_path is provided, use it and any .ply files in objects_path
    if scene_path:
        splatviz = Splatviz(data_path=data_path, mode=mode, host=host, port=port,
                                scene_path=scene_path, objects_path=objects_path, rotation=rotation, caminv=caminv, jsonfile=cfg)
    else:
        splatviz = Splatviz(data_path=data_path, mode=mode, host=host, port=port)



    if objects_path:
        load_dynamic_objects(Path(objects_path), splatviz, world_quat)

    # model_ply = checkpoint_path + "point_cloud/iteration_60000/point_cloud.ply"
    # id_module_ckpt = checkpoint_path + "id_module.th"
    # predictor = None
    # if os.path.isfile(model_ply) and os.path.isfile(id_module_ckpt):
    #     predictor = SixDGSPredictor(model_ply, id_module_ckpt)
    # else:
    #     print("Warning: 6DGS model or weights path not provided. External pose prediction disabled.")

    # 영상 입력 초기화 (웹캠 또는 비디오 파일)
    # cap = None
    # if predictor is not None:
    #     if video == "":
    #         cap_index = 0
    #         cap = cv2.VideoCapture(cap_index)
    #     else:
    #         # video가 숫자 문자열이면 웹캠 인덱스로, 아니면 파일 경로로 처리
    #         if video.isdigit():
    #             cap_index = int(video)
    #             cap = cv2.VideoCapture(cap_index)
    #         else:
    #             cap = cv2.VideoCapture(video)
    #     if not cap or not cap.isOpened():
    #         print(f"Error: cannot open video source `{video}`")
    #         cap = None

    # # 별도 쓰레드로 카메라 프레임 캡처 및 포즈 예측
    # from collections import deque
    #
    # # -------- 공유 버퍼 ----------
    # frame_buf = deque(maxlen=1)  # 길이 1 → 항상 '가장 최신' 프레임만 보관
    # pose_buf = dict(mat=None, new=False)
    # buf_lock = threading.Lock()
    # stop_evt = threading.Event()
    #
    # def capture_loop():
    #     while not stop_evt.is_set():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break  # EOF or error
    #         with buf_lock:
    #             frame_buf.append(frame)  # 최신 프레임 덮어쓰기
    #         cv2.imshow("Video", frame)  # 실시간 표시
    #         if cv2.waitKey(1) & 0xFF == 27:  # ESC
    #             stop_evt.set()
    #             break
    #
    # def predict_loop():
    #     last_seen_id = 0
    #     while not stop_evt.is_set():
    #         with buf_lock:
    #             if not frame_buf:  # 아직 프레임 없음
    #                 continue
    #             frame = frame_buf[-1].copy()  # 가장 최신 프레임 스냅샷
    #             fid = id(frame)  # 객체 ID로 “신규 여부” 판별
    #         if fid == last_seen_id:
    #             time.sleep(0.005)  # 이미 처리한 프레임 → 살짝 대기
    #             continue
    #         last_seen_id = fid
    #
    #         pose = predictor.predict_pose(frame)  # ★ 딥러닝 추론 ★
    #
    #         with buf_lock:
    #             pose_buf["mat"] = pose
    #             pose_buf["new"] = True
    #     stop_evt.set()
    #
    # # 쓰레드 시작
    # if predictor is not None and cap is not None:
    #     # 1) 캡처·디스플레이
    #     cap_th = threading.Thread(target=capture_loop, daemon=True, name="capture")
    #     # 2) 포즈 예측
    #     pred_th = threading.Thread(target=predict_loop, daemon=True, name="predict")
    #
    #     cap_th.start()
    #     pred_th.start()
    # else:
    #     cap_th = None
    #     pred_th = None
    PHYS_STEPS_PER_FRAME = 3
    while not splatviz.should_close():
        for _ in range(PHYS_STEPS_PER_FRAME):
            p.stepSimulation()
        splatviz.sync_dynamic_objects(scene_origin_pos=[0, 0, 0], scene_origin_quat=world_quat)
        # if predictor is not None:
        #     with buf_lock:
        #         if pose_buf["new"]:
        #             splatviz.set_external_camera_pose(pose_buf["mat"])
        #             pose_buf["new"] = False
        splatviz.draw_frame()
    # 루프 종료: 자원 해제 및 쓰레드 정리
    # if cap_th and pred_th is not None:
    #     cap_th.join(timeout=1.0)
    #     pred_th.join(timeout=1.0)
    #     cap.release()
    cv2.destroyAllWindows()
    splatviz.close()

if __name__ == '__main__':
    main()
