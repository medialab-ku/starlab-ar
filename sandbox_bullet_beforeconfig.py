# bullet_room_fit.py  (was: warp_room_sim.py)
# ─────────────────────────────────────────────────────────────
# PyBullet + SPSA only
# - Static room mesh (OBJ) vs. dynamic rigid sphere (from OBJ bounding sphere)
# - One-shot run: generate GT by Bullet → fit (SPSA, 2-stage) → overlay/save
# ─────────────────────────────────────────────────────────────

import argparse, os, sys, math, random
from time import perf_counter
import numpy as np
import trimesh
import open3d as o3d

# PyBullet
try:
    import pybullet as p
    import pybullet_data
    from pybullet_utils import bullet_client
except Exception as _e:
    raise RuntimeError("PyBullet not available. Install with: pip install pybullet") from _e


# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────

def rotx(deg: float):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def load_obj_vertices_faces(path: str):
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"'{path}' 로드 실패(삼각형 메쉬 아님)")
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    return v, f, mesh

def compute_bounding_sphere(verts: np.ndarray, center_hint: str = "com"):
    if center_hint == "origin":
        c = np.zeros(3, dtype=np.float32)
    elif center_hint == "bbox":
        bbmin = verts.min(0); bbmax = verts.max(0)
        c = (bbmin + bbmax) * 0.5
    else:  # "com"
        c = verts.mean(0)
    r = np.linalg.norm(verts - c, axis=1).max()
    return c.astype(np.float32), float(r)

# --- NEW: path-aware bounding sphere util ---
def compute_bounding_sphere_from_obj(path: str, center_hint: str = "com"):
    """
    OBJ 파일 경로를 받아 바운딩 스피어 (center, radius)를 리턴.
    center_hint: "com" | "bbox" | "origin"
    """
    import trimesh
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"'{path}' is not a triangle mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if verts.size == 0:
        # 비어있으면 안전한 기본값
        return np.zeros(3, dtype=np.float32), 0.1
    if center_hint == "origin":
        c = np.zeros(3, dtype=np.float32)
    elif center_hint == "bbox":
        bbmin = verts.min(0); bbmax = verts.max(0)
        c = (bbmin + bbmax) * 0.5
    else:  # "com"
        c = verts.mean(0)
    r = float(np.linalg.norm(verts - c, axis=1).max())
    return c.astype(np.float32), r
# ─────────────────────────────────────────────────────────────
# PyBullet backend
# ─────────────────────────────────────────────────────────────

def build_bullet_cache(scene_obj,
                       object_paths,
                       scene_rotx_deg,
                       start_positions,
                       gravity,
                       dt,
                       substeps,
                       scene_body_id=None):
    """
    현재 연결된 PyBullet 월드를 그대로 쓴다(새 connect/reset 금지).
    - scene_body_id가 주어지면 그 body를 scene으로 간주하고 별도 로딩하지 않는다.
    - 필요한 전역 파라미터만 읽어서 캐시에 담는다.
    """
    # 현재 월드 스냅샷(필요 시 복구용)
    state_id = p.saveState()
    # Pre-compute dynamic-sphere radii from object OBJ paths (needed by SPSA weighting)
    radii = []
    if object_paths is not None:
        for oi in range(min(2, len(object_paths))):
            try:
                _, r = compute_bounding_sphere_from_obj(object_paths[oi])
                r = float(r) * 1.01
            except Exception:
                r = 0.1
            radii.append(float(r))
    # Fallback if missing/None
    if len(radii) == 0:
        radii = [0.1, 0.1]
    # (월드 파라미터만 캐시에 보관)
    cache = dict(
        scene_body_id=scene_body_id,          # None이면 scene 없음으로 간주
        scene_rotx_deg=float(scene_rotx_deg),
        start_positions=[list(start_positions[0]), list(start_positions[1])],
        gravity=tuple(gravity),
        dt=float(dt),
        substeps=int(substeps),
        object_paths=list(object_paths) if object_paths is not None else None,
        radii=radii,
        _state_id=state_id
    )
    return cache

def simulate_bullet_once_multi(args_base,
                               v0_flat,
                               mu, e,
                               mu_roll, mu_spin,
                               masses,
                               seconds_=None,
                               **kwargs):
    # ── 격리 캐시가 있으면 그걸 재사용 (원본과 동일 구조)
    iso = args_base.get("bullet_cache_isolated", None)
    if iso is not None:
        bc = iso["bc"]
        starts = iso["starts"]
        substeps = iso["substeps"]
        dt = iso["dt"]
        seconds = float(seconds_ if seconds_ is not None else args_base.get("seconds", 1.5))

        # 이전 동적 바디가 있으면 제거 (바디 id 저장/초기화)
        if "dyn_ids" in iso:
            for bid in iso["dyn_ids"]:
                try: bc.removeBody(bid)
                except Exception: pass
        iso["dyn_ids"] = []

        # sphere_shapes 재사용 → mass 반영하여 바디 생성
        dyn_ids = []
        N = 2
        for i in range(N):
            col_id, vis_id, r = iso["sphere_shapes"][i]
            bid = bc.createMultiBody(
                baseMass=float(masses[i]),
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=starts[i],
                baseOrientation=[0,0,0,1]
            )
            bc.changeDynamics(
                bid, -1,
                lateralFriction=float(mu),
                restitution=float(e),
                rollingFriction=float(mu_roll),
                spinningFriction=float(mu_spin),
                frictionAnchor=1,
                linearDamping=0.0,
                angularDamping=0.0
            )
            vx,vy,vz = list(map(float, v0_flat[3*i:3*i+3]))
            bc.resetBaseVelocity(bid, [vx,vy,vz])
            dyn_ids.append(bid)

        iso["dyn_ids"] = dyn_ids

        steps = int(np.ceil(seconds / float(dt)))
        traj = np.zeros((steps+1, N, 3), dtype=np.float32)
        for i, bid in enumerate(dyn_ids):
            pos,_ = bc.getBasePositionAndOrientation(bid)
            traj[0,i,:] = np.asarray(pos, np.float32)

        for t in range(1, steps+1):
            for _ in range(int(substeps)):
                bc.stepSimulation()
            for i, bid in enumerate(dyn_ids):
                pos,_ = bc.getBasePositionAndOrientation(bid)
                traj[t,i,:] = np.asarray(pos, np.float32)

        # 동적 바디는 남겨두거나 제거 둘 중 하나 선택:
        # - 남겨두면 다음 iter에서 재사용/제거하기 쉬움
        # 여기서는 남겨두되, 다음 iter 시작 시 제거
        return traj
    # 캐시에서 파라미터
    scene_body_id = args_base.get("scene_body_id", None)
    dt = args_base["dt"]
    substeps = args_base["substeps"]
    gravity = args_base["gravity"]
    starts = args_base["start_positions"]
    seconds = float(seconds_ if seconds_ is not None else args_base["seconds"])

    # v0_flat -> [[...],[...]]
    v0s = [ list(map(float, v0_flat[:3])), list(map(float, v0_flat[3:6])) ]

    # === 스냅샷
    state_id = p.saveState()
    try:
        try: p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        except: pass

        p.setGravity(*gravity)
        h = float(dt) / max(1, int(substeps))
        p.setTimeStep(h)
        try:
            p.setPhysicsEngineParameter(
                numSolverIterations=50,
                useSplitImpulse=1,
                splitImpulsePenetrationThreshold=-0.02,
                deterministicOverlappingPairs=1,
                restitutionVelocityThreshold=0.0
            )
        except Exception:
            pass

        # 임시 동적체 스폰
        dyn_ids = []
        radii = []
        # object_paths는 캐시에 있긴 하지만, 구체 반지름은 캐시 외부에서 재계산해도 무방
        obj_paths = args_base.get("object_paths", [None, None])
        for oi in range(2):
            if obj_paths[oi]:
                _, r = compute_bounding_sphere_from_obj(obj_paths[oi])
            else:
                r = 0.1
            r = float(r) * 1.01
            radii.append(r)
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=r)
            bid = p.createMultiBody(baseMass=float(masses[oi]),
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=starts[oi],
                                    baseOrientation=[0,0,0,1])
            dyn_ids.append(bid)
            p.resetBaseVelocity(bid, linearVelocity=v0s[oi])

            try:
                p.changeDynamics(
                    bid, -1,
                    lateralFriction=float(mu),
                    restitution=float(e),
                    rollingFriction=float(mu_roll),
                    spinningFriction=float(mu_spin),
                    frictionAnchor=1,
                    linearDamping=0.0,
                    angularDamping=0.0,
                )
            except Exception:
                pass

        steps = int(np.ceil(seconds / float(dt)))
        traj = np.zeros((steps+1, 2, 3), dtype=np.float32)
        for oi, bid in enumerate(dyn_ids):
            pos, orn = p.getBasePositionAndOrientation(bid)
            traj[0, oi] = np.array(pos, dtype=np.float32)

        for t in range(1, steps+1):
            for _ in range(int(substeps)):
                p.stepSimulation()
            for oi, bid in enumerate(dyn_ids):
                pos, orn = p.getBasePositionAndOrientation(bid)
                traj[t, oi] = np.array(pos, dtype=np.float32)

        for bid in dyn_ids:
            p.removeBody(bid)

        return traj
    finally:
        try: p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        except: pass
        p.restoreState(state_id)
        p.removeState(state_id)

def bullet_record_traj_multi(scene_path,
                             object_paths,
                             seconds,
                             dt,
                             substeps,
                             scene_rotx_deg,
                             gravity,
                             starts,
                             v0s,
                             mu, e,
                             mu_roll, mu_spin,
                             masses,
                             scene_body_id=None):
    """
    현재 월드에서 임시 동적체만 생성해 궤적을 기록하고,
    끝나면 모두 지우고 saveState로 복구한다.
    scene_body_id가 주어지면 그 body를 지형으로 사용(추가 로딩 없음).
    """
    # === 스냅샷 저장
    state_id = p.saveState()
    # 렌더 잠깐 끄기(빠르게)
    try: p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    except: pass

    try:
        # 중력/타임스텝만 설정(restore로 원복됨)
        p.setGravity(*gravity)
        h = float(dt) / max(1, int(substeps))
        p.setTimeStep(h)
        try:
            p.setPhysicsEngineParameter(
                numSolverIterations=50,
                useSplitImpulse=1,
                splitImpulsePenetrationThreshold=-0.02,
                deterministicOverlappingPairs=1,
                restitutionVelocityThreshold=0.0
            )
        except Exception:
            pass
        # scene은 이미 월드에 있다고 가정(없으면 그냥 빈 공간)
        # 임시 동적체(구체) 생성
        dyn_ids = []
        radii = []
        for oi in range(2):
            # obj 바운딩 스피어 반지름 계산(기존 구현 사용)
            _, r = compute_bounding_sphere_from_obj(object_paths[oi])
            r = float(r) * 1.01
            radii.append(r)
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=r)
            bid = p.createMultiBody(baseMass=float(masses[oi]),
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=starts[oi],
                                    baseOrientation=[0,0,0,1])
            dyn_ids.append(bid)
            # 초기 속도
            try:
                p.changeDynamics(
                    bid, -1,
                    lateralFriction=float(mu),
                    restitution=float(e),
                    rollingFriction=float(mu_roll),
                    spinningFriction=float(mu_spin),
                    frictionAnchor=1,
                    linearDamping=0.0,
                    angularDamping=0.0,
                )
            except Exception:
                pass
            p.resetBaseVelocity(bid, linearVelocity=v0s[oi])

        steps = int(np.ceil(seconds / float(dt)))
        traj = np.zeros((steps+1, 2, 3), dtype=np.float32)
        # 기록 초기값
        for oi, bid in enumerate(dyn_ids):
            pos, orn = p.getBasePositionAndOrientation(bid)
            traj[0, oi] = np.array(pos, dtype=np.float32)

        # 시뮬 진행
        for t in range(1, steps+1):
            # 서브스텝
            for _ in range(int(substeps)):
                p.stepSimulation()
            # 기록
            for oi, bid in enumerate(dyn_ids):
                pos, orn = p.getBasePositionAndOrientation(bid)
                traj[t, oi] = np.array(pos, dtype=np.float32)

        # 임시 바디 정리
        for bid in dyn_ids:
            p.removeBody(bid)
        return traj
    finally:
        # 렌더 on
        try: p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        except: pass
        # 스냅샷 복구(월드 완전 원복)
        p.restoreState(state_id)
        p.removeState(state_id)

def simulate_bullet_once_multi_isolated(args_base,
                                        v0_flat,
                                        mu, e,
                                        mu_roll, mu_spin,
                                        masses,
                                        seconds_=None):
    # ↓ 전역 p가 아니라, 별도의 BulletClient 인스턴스를 생성
    bc = bullet_client.BulletClient(connection_mode=p.DIRECT)
    try:
        import numpy as np, math, trimesh, pybullet_data
        dt        = float(args_base["dt"])
        substeps  = int(args_base["substeps"])
        gravity   = tuple(args_base["gravity"])
        starts    = args_base["start_positions"]
        seconds   = float(seconds_ if seconds_ is not None else args_base.get("seconds", 1.5))
        scene_deg = float(args_base.get("scene_rotx_deg", 0.0))
        obj_paths = list(args_base.get("object_paths", [None, None]))
        scene_obj = args_base.get("scene_mesh_path", None)

        bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        bc.resetSimulation()
        bc.setGravity(*gravity)

        h = dt / max(1, substeps)
        bc.setTimeStep(h)
        try:
            bc.setPhysicsEngineParameter(
                numSolverIterations=50,
                useSplitImpulse=1,
                splitImpulsePenetrationThreshold=-0.02,
                deterministicOverlappingPairs=1,
                restitutionVelocityThreshold=0.0
            )
        except Exception:
            pass

        # Scene concave mesh (격리 월드 내부에만 로드)
        if scene_obj is not None:
            flags = p.GEOM_FORCE_CONCAVE_TRIMESH
            sc_col = bc.createCollisionShape(p.GEOM_MESH, fileName=scene_obj, flags=flags)
            sc_vis = bc.createVisualShape(p.GEOM_MESH, fileName=scene_obj)
            rx = math.radians(scene_deg)
            sc_quat = bc.getQuaternionFromEuler([rx, 0.0, 0.0])
            _scene_id = bc.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=sc_col,
                baseVisualShapeIndex=sc_vis,
                basePosition=[0,0,0],
                baseOrientation=sc_quat
            )

        # Dynamic spheres (반지름 1.01배)
        body_ids = []
        for i in range(2):
            if obj_paths[i]:
                _, r = compute_bounding_sphere_from_obj(obj_paths[i])
            else:
                r = 0.1
            r = float(r) * 1.01
            col = bc.createCollisionShape(p.GEOM_SPHERE, radius=r)
            vis = bc.createVisualShape(p.GEOM_SPHERE, radius=r)
            bid = bc.createMultiBody(baseMass=float(masses[i]),
                                     baseCollisionShapeIndex=col,
                                     baseVisualShapeIndex=vis,
                                     basePosition=starts[i],
                                     baseOrientation=[0,0,0,1])
            bc.changeDynamics(
                bid, -1,
                lateralFriction=float(mu),
                restitution=float(e),
                rollingFriction=float(mu_roll),
                spinningFriction=float(mu_spin),
                frictionAnchor=1,
                linearDamping=0.0,
                angularDamping=0.0
            )
            vx, vy, vz = list(map(float, v0_flat[3*i:3*i+3]))
            bc.resetBaseVelocity(bid, [vx, vy, vz])
            body_ids.append(bid)

        steps = int(np.ceil(seconds / dt))
        traj = np.zeros((steps+1, 2, 3), dtype=np.float32)
        for i, bid in enumerate(body_ids):
            pos,_ = bc.getBasePositionAndOrientation(bid)
            traj[0,i,:] = np.asarray(pos, np.float32)

        for t in range(1, steps+1):
            for _ in range(substeps):
                bc.stepSimulation()
            for i, bid in enumerate(body_ids):
                pos,_ = bc.getBasePositionAndOrientation(bid)
                traj[t,i,:] = np.asarray(pos, np.float32)

        return traj
    finally:
        # ← 여기서 전역 p.disconnect()가 아니라, 이 격리 client만 끊음
        bc.disconnect()

def build_bullet_cache_isolated(scene_obj,
                                object_paths,
                                scene_rotx_deg,
                                start_positions,
                                gravity,
                                dt,
                                substeps):
    """
    원본처럼: DIRECT BulletClient 1회 생성 → scene concave mesh 1회 로드 →
    동적 구체 shape 1회 생성. 이 캐시를 Stage-1/2 내내 재사용.
    """
    bc = bullet_client.BulletClient(connection_mode=p.DIRECT)
    import numpy as np, math, pybullet_data

    bc.setAdditionalSearchPath(pybullet_data.getDataPath())
    bc.resetSimulation()
    bc.setGravity(*gravity)

    h = float(dt) / max(1, int(substeps))
    bc.setTimeStep(h)
    try:
        bc.setPhysicsEngineParameter(
            numSolverIterations=50,
            useSplitImpulse=1,
            splitImpulsePenetrationThreshold=-0.02,
            deterministicOverlappingPairs=1,
            restitutionVelocityThreshold=0.0
        )
    except Exception:
        pass

    # ── 정적 씬(원본과 동일하게 concave mesh + 회전 적용)
    scene_id = None
    scene_quat = None
    if scene_obj is not None:
        flags = p.GEOM_FORCE_CONCAVE_TRIMESH
        sc_col = bc.createCollisionShape(p.GEOM_MESH, fileName=scene_obj, flags=flags)
        sc_vis = bc.createVisualShape(p.GEOM_MESH, fileName=scene_obj)
        rx = math.radians(float(scene_rotx_deg))
        scene_quat = bc.getQuaternionFromEuler([rx, 0.0, 0.0])
        scene_id = bc.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=sc_col,
            baseVisualShapeIndex=sc_vis,
            basePosition=[0,0,0],
            baseOrientation=scene_quat
        )

    # ── 동적 구체 shape 1회 생성 (반지름 1.01배; 원본과 동일)
    sphere_shapes = []  # (col_id, vis_id, radius)
    radii = []
    for i in range(2):
        _, r = compute_bounding_sphere_from_obj(object_paths[i])
        r = float(r) * 1.01
        radii.append(r)
        col = bc.createCollisionShape(p.GEOM_SPHERE, radius=r)
        vis = bc.createVisualShape(p.GEOM_SPHERE, radius=r)
        sphere_shapes.append((col, vis, r))

    return dict(
        bc=bc,
        scene_id=scene_id,
        scene_quat=scene_quat,
        sphere_shapes=sphere_shapes,
        radii=radii,
        starts=[list(start_positions[0]), list(start_positions[1])],
        dt=float(dt),
        substeps=int(substeps),
        scene_rotx_deg=float(scene_rotx_deg),
        gravity=tuple(gravity),
        object_paths=list(object_paths)
    )

def destroy_bullet_cache_isolated(cache):
    try:
        cache["bc"].disconnect()
    except Exception:
        pass
# ─────────────────────────────────────────────────────────────
# Loss / Opt / Viz
# ─────────────────────────────────────────────────────────────

def traj_loss(traj_pred, traj_gt, stride=1, weights=None,
              reduction="mean",    # "mean" | "sum"
              p="l2",              # "l2" | "l1" | "charbonnier"
              epsilon=1e-3,
              time_gamma=None):    # None 또는 >=0 (뒤로 갈수록 가중↑)
    """
    일반화된 궤적 손실:
      - reduction: "mean"(기본) 또는 "sum"
      - p: L2(제곱), L1(절대), Charbonnier(√(x^2+eps^2))
      - time_gamma: 프레임 t의 가중을 (t/T)^gamma 로 증가 (누적오차 민감)
    """
    P = np.asarray(traj_pred)
    G = np.asarray(traj_gt)

    # [T+1,N,3] 정렬 + stride
    T1 = min(P.shape[0], G.shape[0])
    P = P[:T1][::stride]
    G = G[:T1][::stride]

    # 프레임별 오차(바디/축 평균)
    D = P - G                    # [Tf,N,3]
    if p == "l2":
        per_frame = np.mean(np.sum(D*D, axis=2), axis=1)    # [Tf]
    elif p == "l1":
        per_frame = np.mean(np.sum(np.abs(D), axis=2), axis=1)
    elif p == "charbonnier":
        per_frame = np.mean(np.sqrt(np.sum(D*D, axis=2) + epsilon**2), axis=1)
    else:
        raise ValueError("p must be l2|l1|charbonnier")

    # 시간가중 (뒤로 갈수록↑) : 누적오차 민감도 상승
    if time_gamma is not None and T1 > 1:
        Tf = per_frame.shape[0]
        t = np.arange(Tf, dtype=np.float32)
        wt = ((t + 1e-6) / (Tf - 1 + 1e-6)) ** float(time_gamma)
        per_frame = per_frame * wt

    # 외부 프레임 가중치와 병합 (normalize to mean ~1)
    if weights is not None:
        W = np.asarray(weights, dtype=np.float32)
        W = W[:T1][::stride]
        W = W / (np.mean(W) + 1e-8)
        per_frame = per_frame * W

    if reduction == "sum":
        return float(np.sum(per_frame))
    else:
        return float(np.mean(per_frame))

def bodywise_rms(traj_pred, traj_gt, stride=1, weights=None):
    """바디별 RMS 오차 반환: shape (N,)"""
    P = np.asarray(traj_pred); G = np.asarray(traj_gt)
    T1 = min(P.shape[0], G.shape[0])
    P = P[:T1][::stride]; G = G[:T1][::stride]        # [Tf,N,3]
    D = P - G                                         # [Tf,N,3]
    per_frame_body = np.sqrt(np.sum(D*D, axis=2))     # [Tf,N]
    if weights is not None:
        W = np.asarray(weights, dtype=np.float32)
        W = W[:T1][::stride]
        W = W / (np.mean(W) + 1e-8)
        per_frame_body = per_frame_body * W[:, None]
    # 프레임 평균 → 바디별 RMS
    return np.mean(per_frame_body, axis=0)            # [N]

def compute_collision_weights(traj_gt: np.ndarray, dt: float, smooth=5, boost=3.0):
    """GT 궤적에서 가속도 스파이크(충돌/구름 전환)를 찾아 가중치 ↑
       - 간단한 2차 차분 기반
       - smooth: 이동평균 길이
       - boost: 충돌 프레임 가중 확대 배수
    """
    G = traj_gt[:,0,:] if traj_gt.ndim==3 else traj_gt  # 대략 첫 바디 기준; 여러 바디면 평균 사용 가능
    if traj_gt.ndim==3:
        G = np.mean(traj_gt, axis=1)  # [T+1,3] 바디 평균으로 더 robust
    vel = (G[1:] - G[:-1]) / dt                    # [T,3]
    acc = (vel[1:] - vel[:-1]) / dt                # [T-1,3]
    a  = np.linalg.norm(acc, axis=1)               # [T-1]
    # 패딩
    a = np.concatenate([[a[0]], a, [a[-1]]], axis=0)  # [T+1-?] 맞춤
    # 이동평균 스무딩
    if smooth > 1:
        k = np.ones(smooth, dtype=np.float32)/smooth
        a = np.convolve(a, k, mode="same")
    # 정규화 후 충돌 프레임 부스트
    a = a / (np.mean(a) + 1e-8)
    w = 1.0 + (boost-1.0) * (a / (np.max(a)+1e-8))
    return w.astype(np.float32)  # 길이 ≈ T+1

def find_first_pair_contact(traj_gt: np.ndarray, radii: list, eps=0.02):
    """GT 궤적에서 첫 동적-동적 접촉 프레임 인덱스(k*)를 반환. 없으면 -1."""
    G = np.asarray(traj_gt)  # [T+1,N,3]
    T1, N = G.shape[0], G.shape[1]
    if N < 2:
        return -1
    for t in range(T1):
        for i in range(N):
            for j in range(i+1, N):
                dij = np.linalg.norm(G[t, i] - G[t, j])
                if dij <= (radii[i] + radii[j]) * (1.0 + eps):
                    return t
    return -1

def vel_loss_pair_normal(traj_pred: np.ndarray, traj_gt: np.ndarray, k_contact: int, dt: float, window: int = 4):
    """첫 접촉 프레임 k_contact 주변에서 두 물체 상대 법선속도 오차를 L2로 계산.
       - pred가 fast_seconds로 짧을 수 있으므로, 내부에서 길이 정렬/클램프 처리
    """
    P_full = np.asarray(traj_pred)  # [Tp1_pred, N, 3]
    G_full = np.asarray(traj_gt)    # [Tp1_gt,   N, 3]
    if P_full.ndim != 3 or G_full.ndim != 3:
        return 0.0
    Np = P_full.shape[1]
    Ng = G_full.shape[1]
    if min(Np, Ng) < 2:
        return 0.0

    # 공통 길이로 정렬
    Tp1 = min(P_full.shape[0], G_full.shape[0])
    P = P_full[:Tp1]
    G = G_full[:Tp1]

    # k_contact 클램프 (양쪽 모두에서 접근 가능하도록)
    if k_contact < 1 or k_contact >= Tp1-1:
        # 접촉이 너무 초반/후반이면 비교를 생략
        return 0.0

    def rel_nvel(traj):
        # 속도: 중앙차분 (경계는 전/후진차분)
        V = np.zeros_like(traj)
        V[1:-1] = (traj[2:] - traj[:-2]) / (2.0*dt)
        V[0]    = (traj[1] - traj[0]) / dt
        V[-1]   = (traj[-1] - traj[-2]) / dt
        # 법선: 두 물체(0,1) 사이 방향
        n = traj[:, 1, :] - traj[:, 0, :]
        n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
        nhat = n / n_norm
        vr = V[:, 1, :] - V[:, 0, :]  # 상대속도
        vn = np.sum(vr * nhat, axis=1)  # [Tp1]
        return vn

    vn_p = rel_nvel(P)
    vn_g = rel_nvel(G)

    # 창 클램프: a..b 가 양쪽에 대해 유효하도록
    a = max(0, k_contact - window)
    b = min(Tp1, k_contact + window + 1)
    if b <= a + 1:
        return 0.0

    diff = vn_p[a:b] - vn_g[a:b]  # 길이 동일 보장
    return float(np.mean(diff * diff))

def dvel_loss_window(traj_pred: np.ndarray, traj_gt: np.ndarray, k_contact: int, dt: float, window: int = 2):
    """k_contact 주변에서 각 바디의 속도변화(Δv) L2 오차."""
    P = np.asarray(traj_pred); G = np.asarray(traj_gt)
    if P.ndim != 3 or G.ndim != 3:
        return 0.0
    T1 = min(P.shape[0], G.shape[0])
    P = P[:T1]; G = G[:T1]
    N = G.shape[1]
    if k_contact < 2 or k_contact >= T1-2 or N < 2:
        return 0.0

    def vel(traj):
        V = np.zeros_like(traj)
        V[1:-1] = (traj[2:] - traj[:-2]) / (2.0*dt)
        V[0]    = (traj[1] - traj[0]) / dt
        V[-1]   = (traj[-1] - traj[-2]) / dt
        return V

    Vp, Vg = vel(P), vel(G)
    a = max(1, k_contact - window)
    b = min(T1-1, k_contact + window)
    dVp = Vp[a+1:b+1] - Vp[a-1:b-1]
    dVg = Vg[a+1:b+1] - Vg[a-1:b-1]
    diff = dVp - dVg
    return float(np.mean(diff*diff))

def momentum_normal_loss(traj_pred: np.ndarray, traj_gt: np.ndarray, k_contact: int, dt: float,
                         masses: np.ndarray, window: int = 2):
    """첫 동적-동적 접촉 주변(±window)에서 법선방향 총운동량 변화(전후)의 차이를 벌점."""
    P = np.asarray(traj_pred); G = np.asarray(traj_gt)
    if P.ndim != 3 or G.ndim != 3:
        return 0.0
    T1 = min(P.shape[0], G.shape[0])
    P = P[:T1]; G = G[:T1]
    N = G.shape[1]
    if k_contact < 2 or k_contact >= T1-2 or N < 2:
        return 0.0

    def vel(traj):
        V = np.zeros_like(traj)
        V[1:-1] = (traj[2:] - traj[:-2]) / (2.0*dt)
        V[0]    = (traj[1] - traj[0]) / dt
        V[-1]   = (traj[-1] - traj[-2]) / dt
        return V

    def p_normal(traj, V, masses_):
        # 법선은 0↔1 바디 연결 방향
        n = traj[:, 1, :] - traj[:, 0, :]
        nhat = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)  # [T,1,3]
        # 각 바디 속도의 법선성분
        vn = np.sum(V * nhat[:, None, :], axis=2)   # [T,N]
        mcol = np.asarray(masses_, np.float32)[None, :]  # [1,N]
        p_n = np.sum(mcol * vn, axis=1)             # [T]
        return p_n

    Vp, Vg = vel(P), vel(G)
    a = max(1, k_contact - window)
    b = min(T1-1, k_contact + window)

    pn_pre_p  = np.mean(p_normal(P[a-1:b-1], Vp[a-1:b-1], masses))
    pn_post_p = np.mean(p_normal(P[a+1:b+1], Vp[a+1:b+1], masses))
    # GT는 GT의 실제 질량으로 계산 (gt_traj를 만들 때 썼던 gt_masses)
    # 여기서는 GT 질량 대신 "관측량"으로 보고 변화량만 비교
    Vg_pre  = np.mean(p_normal(G[a-1:b-1], Vg[a-1:b-1], np.ones((N,), np.float32)))
    Vg_post = np.mean(p_normal(G[a+1:b+1], Vg[a+1:b+1], np.ones((N,), np.float32)))
    # GT에서의 법선 운동량 변화량 대비, 예측의 변화량이 같아야 한다고 가정
    dp_p = pn_post_p - pn_pre_p
    dp_g = Vg_post - Vg_pre
    return float((dp_p - dp_g) ** 2)

def e_consistency_loss(traj_pred: np.ndarray, traj_gt: np.ndarray, k_contact: int, dt: float, window: int = 2):
    """첫 동적-동적 접촉 근방(±window)에서, GT로부터 관측한 e와
       예측으로부터 관측한 e의 차이를 L2로 벌점."""
    P = np.asarray(traj_pred); G = np.asarray(traj_gt)
    if P.ndim != 3 or G.ndim != 3:
        return 0.0
    T1 = min(P.shape[0], G.shape[0])
    P = P[:T1]; G = G[:T1]
    if k_contact < 2 or k_contact >= T1-2 or G.shape[1] < 2:
        return 0.0

    def rel_n_v(traj):
        V = np.zeros_like(traj)
        V[1:-1] = (traj[2:] - traj[:-2]) / (2.0*dt)
        V[0]    = (traj[1] - traj[0]) / dt
        V[-1]   = (traj[-1] - traj[-2]) / dt
        n = traj[:, 1, :] - traj[:, 0, :]
        nhat = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
        vr = V[:, 1, :] - V[:, 0, :]
        vn = np.sum(vr * nhat, axis=1)  # [T]
        return vn

    vn_p = rel_n_v(P); vn_g = rel_n_v(G)
    a = max(1, k_contact - window)
    b = min(T1-1, k_contact + window)

    # 전/후 평균으로 노이즈 제거
    pre_p  = np.mean(vn_p[a-1:b-1]); post_p = np.mean(vn_p[a+1:b+1])
    pre_g  = np.mean(vn_g[a-1:b-1]); post_g = np.mean(vn_g[a+1:b+1])

    # 분모가 너무 작으면 무력화
    eps = 1e-5
    if abs(pre_p) < eps or abs(pre_g) < eps:
        return 0.0

    e_pred = -post_p / (pre_p + np.sign(pre_p)*eps)
    e_gt   = -post_g / (pre_g + np.sign(pre_g)*eps)

    # 합리 범위로 클리핑
    e_pred = float(np.clip(e_pred, 0.0, 1.5))
    e_gt   = float(np.clip(e_gt,   0.0, 1.5))
    return (e_pred - e_gt) ** 2

def mass_consistency_loss(traj_gt: np.ndarray, k_contact: int, dt: float,
                          masses: np.ndarray, window: int = 2):
    """GT 궤적에서 관측한 {v1_pre, v2_pre, v1_post, v2_post, e_gt}가
       1D 두-물체 충돌식과 (예측) 질량 m1,m2로 일치하도록 벌점."""
    G = np.asarray(traj_gt)
    if G.ndim != 3 or G.shape[1] < 2:
        return 0.0
    T1 = G.shape[0]
    if k_contact < 2 or k_contact >= T1-2:
        return 0.0

    def rel_n_v(traj):
        V = np.zeros_like(traj)
        V[1:-1] = (traj[2:] - traj[:-2]) / (2.0*dt)
        V[0]    = (traj[1] - traj[0]) / dt
        V[-1]   = (traj[-1] - traj[-2]) / dt
        n = traj[:, 1, :] - traj[:, 0, :]
        nhat = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
        vr = V[:, 1, :] - V[:, 0, :]
        vn = np.sum(vr * nhat, axis=1)  # [T]
        # 각 바디의 법선 성분 (개별)
        v1n = np.sum(V[:, 0, :] * nhat, axis=1)
        v2n = np.sum(V[:, 1, :] * nhat, axis=1)
        return vn, v1n, v2n

    _, v1n, v2n = rel_n_v(G)
    a = max(1, k_contact - window)
    b = min(T1-1, k_contact + window)

    # 전/후 평균(노이즈 완화)
    v1_pre  = float(np.mean(v1n[a-1:b-1]))
    v2_pre  = float(np.mean(v2n[a-1:b-1]))
    v1_post = float(np.mean(v1n[a+1:b+1]))
    v2_post = float(np.mean(v2n[a+1:b+1]))

    # e_gt 관측
    eps = 1e-6
    vn_pre  = float(np.mean((v2n[a-1:b-1] - v1n[a-1:b-1])))
    vn_post = float(np.mean((v2n[a+1:b+1] - v1n[a+1:b+1])))
    if abs(vn_pre) < 1e-5:
        return 0.0
    e_gt = -vn_post / (vn_pre + np.sign(vn_pre)*eps)
    e_gt = float(np.clip(e_gt, 0.0, 1.5))

    m1, m2 = float(masses[0]), float(masses[1])
    denom  = m1 + m2
    if denom <= eps:
        return 0.0

    # 1D 충돌 방정식으로 예측되는 post
    v1_post_hat = ((m1 - e_gt*m2)*v1_pre + (1.0 + e_gt)*m2*v2_pre) / denom
    v2_post_hat = ((m2 - e_gt*m1)*v2_pre + (1.0 + e_gt)*m1*v1_pre) / denom

    r1 = v1_post - v1_post_hat
    r2 = v2_post - v2_post_hat
    return float(0.5*(r1*r1 + r2*r2))

def compute_pair_contact_weights(traj_gt: np.ndarray, radii: list, smooth=3, boost=3.0, eps=0.02):
    """물체-물체(동적-동적) 접촉 프레임을 강조하는 가중치 (길이 T+1)"""
    G = np.asarray(traj_gt)  # [T+1,N,3]
    Tp1, N = G.shape[0], G.shape[1]
    if N < 2:
        return np.ones((Tp1,), np.float32)

    # 모든 쌍에 대해 중심거리 - (r_i + r_j)
    d_min = np.full((Tp1,), np.inf, np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            dij = np.linalg.norm(G[:, i, :] - G[:, j, :], axis=1)  # [T+1]
            gap = dij - (radii[i] + radii[j]) * (1.0 + eps)
            d_min = np.minimum(d_min, gap.astype(np.float32))

    # gap <= 0 인 근방을 강조
    w = np.ones((Tp1,), np.float32)
    contact = (d_min <= 0.0).astype(np.float32)
    # 이동평균으로 살짝 확장
    if smooth > 1:
        k = np.ones(smooth, dtype=np.float32) / smooth
        contact = np.convolve(contact, k, mode="same")
    w += (boost - 1.0) * (contact / (contact.max() + 1e-8))
    return w.astype(np.float32)

def compute_floor_weights(traj_gt: np.ndarray, gravity: tuple, smooth=5, boost=2.0, q=0.15):
    """바닥/면 접촉(구름/튕김) 프레임을 강조하는 가중치 (길이 T+1)
       - 중력 방향으로의 '고도'가 하위 q-분위수 근방이면 가중 ↑
    """
    G = np.asarray(traj_gt)  # [T+1,N,3] or [T+1,3]
    if G.ndim == 2:
        G = G[:, None, :]  # [T+1,1,3]
    g = np.asarray(gravity, np.float32)
    ghat = g / (np.linalg.norm(g) + 1e-8)

    # 중력 방향 성분(고도)
    alt = np.tensordot(G, ghat, axes=([2], [0]))  # [T+1,N]
    alt = np.mean(alt, axis=1)  # 바디 평균 [T+1]

    # 하위 q 분위수 근방 => 바닥/면 근접
    thr = np.quantile(alt, q)
    near = (alt <= thr).astype(np.float32)
    if smooth > 1:
        k = np.ones(smooth, dtype=np.float32) / smooth
        near = np.convolve(near, k, mode="same")

    w = np.ones_like(alt, np.float32)
    w += (boost - 1.0) * (near / (near.max() + 1e-8))
    return w

def clamp_params(theta, N, v_bounds=None,
                 mu_bounds=(0.2,0.6), e_bounds=(0.1,0.6),
                 mu_roll_bounds=(0.0,0.2), mu_spin_bounds=(0.0,0.2),
                 m_bounds=(0.05,20.0)):
    """theta = [v0(3N), mu, e, mu_roll, mu_spin, masses(N)]"""
    t = np.array(theta, dtype=np.float32).copy()

    # v clamp
    if v_bounds is not None:
        for i in range(N):
            vx, vy, vz = t[3*i:3*i+3]
            (vx_min,vx_max), (vy_min,vy_max), (vz_min,vz_max) = v_bounds
            t[3*i:3*i+3] = [
                float(np.clip(vx, vx_min, vx_max)),
                float(np.clip(vy, vy_min, vy_max)),
                float(np.clip(vz, vz_min, vz_max))
            ]

    # scalar frictions & restitution
    mu_idx      = 3*N
    e_idx       = 3*N+1
    mu_roll_idx = 3*N+2
    mu_spin_idx = 3*N+3

    t[mu_idx]      = float(np.clip(t[mu_idx],      *mu_bounds))
    t[e_idx]       = float(np.clip(t[e_idx],       *e_bounds))
    t[mu_roll_idx] = float(np.clip(t[mu_roll_idx], *mu_roll_bounds))
    t[mu_spin_idx] = float(np.clip(t[mu_spin_idx], *mu_spin_bounds))

    # masses
    for i in range(N):
        mi = 3*N + 4 + i
        t[mi] = float(np.clip(t[mi], *m_bounds))
    return t

def split_params(theta, N):
    """→ (v0_flat(3N,), mu, e, mu_roll, mu_spin, masses(N,))"""
    t = np.array(theta, dtype=np.float32)
    v0_flat = t[:3*N]
    mu      = float(t[3*N])
    e       = float(t[3*N+1])
    mu_roll = float(t[3*N+2])
    mu_spin = float(t[3*N+3])
    masses  = t[3*N+4:3*N+4+N]
    return v0_flat, mu, e, mu_roll, mu_spin, masses

def spsa_optimize(gt_traj, args_base, theta0, N,
                  iters=12, a0=0.08, c0=0.15, Ak=3.0, alpha=0.602, gamma=0.101,
                  stride=2, fast_seconds=0.5, store_k=10,
                  update_scale=None, v_bounds=None):
    """SPSA with cached Bullet world (multi-body). theta = [3N, mu, e, mu_roll, mu_spin, masses(N)].
       변경점:
       - 마지막 반복은 반드시 히스토리에 포함
       - 루프 동안 '최저 검증손실 θ'를 추적하여 최종(초록) 궤적으로 사용
    """
    win_contact = int(args_base.get("contact_window", 3))
    rng = random.Random(0xC0FFEE)
    theta = clamp_params(theta0, N, v_bounds=v_bounds)

    # 기반 프레임 가중(충돌/전이/바닥) -------------------------------------------------
    w_base  = compute_collision_weights(gt_traj, args_base["dt"], smooth=9, boost=3.0)
    radii   = args_base["bullet_cache"]["radii"]
    w_oo    = compute_pair_contact_weights(gt_traj, radii, smooth=3, boost=3.0, eps=0.02)
    w_floor = compute_floor_weights(gt_traj, args_base["gravity"], smooth=7, boost=2.0, q=0.15)

    beta_oo, beta_floor = 3.0, 1.2
    weights_all = w_base + beta_oo * w_oo + beta_floor * w_floor

    # 간단 prior
    mu_prior, e_prior = 0.35, 0.25
    mu_roll_prior, mu_spin_prior = 0.02, 0.01
    lam_mu, lam_e = 1e-2, 1e-2
    lam_roll, lam_spin = 5e-3, 5e-3

    upd = np.ones_like(theta, dtype=np.float32) if update_scale is None else np.array(update_scale, dtype=np.float32)
    hist = []

    # 접촉 프레임 및 시뮬 길이 보정 ----------------------------------------------
    k_star = find_first_pair_contact(gt_traj, args_base["bullet_cache"]["radii"], eps=0.02)
    T_fast = fast_seconds if fast_seconds is not None else args_base["seconds"]
    if k_star != -1:
        needed = min(args_base["seconds"], (k_star + 3) * args_base["dt"])
        T_fast = max(T_fast, needed)

    # 커리큘럼(초기 질량 위주 → 후기 e/μ 점증)
    K_mass = int(0.6 * iters)

    def phase_alphas(k):
        if k < K_mass:
            return dict(
                alpha_pos=0.2, alpha_vn=0.5, alpha_dv=0.7, alpha_mom=0.7,
                alpha_e=0.0, alpha_mratio=1.0
            )
        else:
            t = (k - K_mass) / max(1, iters - K_mass - 1)
            return dict(
                alpha_pos=0.2 + 0.8 * t, alpha_vn=0.5, alpha_dv=0.6, alpha_mom=0.6,
                alpha_e=0.3 + 0.5 * t, alpha_mratio=0.8
            )

    def phase_weights(k, weights_all, k_contact, T_total):
        if k_contact == -1:
            return weights_all
        if k < K_mass:
            w = np.zeros_like(weights_all, dtype=np.float32)
            a = max(0, k_contact - 4)
            tail_frames = int(float(args_base["post_tail"]) / float(args_base["dt"]))
            b = min(len(w), k_contact + 1 + max(1, tail_frames))
            w[a:b] = 1.0
            return w
        else:
            return weights_all

    # warmup run (메모리 풀업)
    v0, mu, e, mu_roll, mu_spin, masses = split_params(theta, N)
    _ = simulate_bullet_once_multi(args_base, v0, mu, e, mu_roll, mu_spin, masses, seconds=T_fast)

    # --------- 추가: '최저 손실 θ' 추적용 변수 ----------
    best_theta = theta.copy()
    best_loss  = float("inf")

    for k in range(iters):
        ck = c0 / ((k + 1.0) ** gamma)
        ak = a0 / ((k + 1.0 + Ak) ** alpha)
        delta = np.array([1 if rng.random() < 0.5 else -1 for _ in range(len(theta))], dtype=np.float32)

        thetap = clamp_params(theta + ck * delta, N, v_bounds=v_bounds)
        thetam = clamp_params(theta - ck * delta, N, v_bounds=v_bounds)

        v0p, mup, ep, mu_rollp, mu_spinp, mp = split_params(thetap, N)
        v0m, mum, em, mu_rollm, mu_spinm, mm = split_params(thetam, N)

        traj_p = simulate_bullet_once_multi(args_base, v0p, mup, ep, mu_rollp, mu_spinp, mp, seconds=T_fast)
        traj_m = simulate_bullet_once_multi(args_base, v0m, mum, em, mu_rollm, mu_spinm, mm, seconds=T_fast)

        A  = phase_alphas(k)
        Wk = phase_weights(k, weights_all, k_star, T_total=len(weights_all))

        # 기본 위치/시간가중 손실 (sum: 후반 누적오차 민감)
        fp = A["alpha_pos"] * traj_loss(traj_p, gt_traj, stride=stride, weights=Wk,
                                        reduction="sum", p="l2",
                                        time_gamma=(1.5 if k >= K_mass else None))
        fm = A["alpha_pos"] * traj_loss(traj_m, gt_traj, stride=stride, weights=Wk,
                                        reduction="sum", p="l2",
                                        time_gamma=(1.5 if k >= K_mass else None))

        # if k_star != -1:
        #     fp += A["alpha_dv"]  * dvel_loss_window(traj_p, gt_traj, k_contact=k_star, dt=args_base["dt"], window=win_contact)
        #     fm += A["alpha_dv"]  * dvel_loss_window(traj_m, gt_traj, k_contact=k_star, dt=args_base["dt"], window=win_contact)
        #
        #     fp += A["alpha_mom"] * momentum_normal_loss(traj_p, gt_traj, k_contact=k_star, dt=args_base["dt"], masses=mp, window=win_contact)
        #     fm += A["alpha_mom"] * momentum_normal_loss(traj_m, gt_traj, k_contact=k_star, dt=args_base["dt"], masses=mm, window=win_contact)
        #
        #     if A["alpha_e"] > 0.0:
        #         fp += A["alpha_e"] * e_consistency_loss(traj_p, gt_traj, k_contact=k_star, dt=args_base["dt"], window=win_contact)
        #         fm += A["alpha_e"] * e_consistency_loss(traj_m, gt_traj, k_contact=k_star, dt=args_base["dt"], window=win_contact)
        #
        #     fp += A["alpha_mratio"] * mass_consistency_loss(gt_traj, k_contact=k_star, dt=args_base["dt"], masses=mp, window=win_contact)
        #     fm += A["alpha_mratio"] * mass_consistency_loss(gt_traj, k_contact=k_star, dt=args_base["dt"], masses=mm, window=win_contact)

        # prior penalty
        mu_pp, e_pp = thetap[3 * N], thetap[3 * N + 1]
        mu_mm, e_mm = thetam[3 * N], thetam[3 * N + 1]
        _, _, _, mu_roll_pp, mu_spin_pp, _ = split_params(thetap, N)
        _, _, _, mu_roll_mm, mu_spin_mm, _ = split_params(thetam, N)
        fp += lam_mu*(mu_pp - mu_prior)**2 + lam_e*(e_pp - e_prior)**2 + lam_roll*(mu_roll_pp - mu_roll_prior)**2 + lam_spin*(mu_spin_pp - mu_spin_prior)**2
        fm += lam_mu*(mu_mm - mu_prior)**2 + lam_e*(e_mm - e_prior)**2 + lam_roll*(mu_roll_mm - mu_roll_prior)**2 + lam_spin*(mu_spin_mm - mu_spin_prior)**2

        # SPSA gradient & update
        ghat  = (fp - fm) / (2.0 * ck) * (1.0 / delta)
        theta = clamp_params(theta - ak * (upd * ghat), N, v_bounds=v_bounds)

        # ---------- 변경점 ①: 히스토리 저장 (마지막 반복은 반드시 저장) ----------
        if (k + 1) == iters or len(hist) < store_k:
            v0c, muc, ec, mu_rollc, mu_spinc, mc = split_params(theta, N)
            traj_now = simulate_bullet_once_multi(args_base, v0c, muc, ec, mu_rollc, mu_spinc, mc, seconds=T_fast)
            hist.append((k, traj_now))

        # ---------- 변경점 ②: '현재 θ' 검증손실로 best 갱신 ----------
        # 같은 창/가중으로 간단 검증
        v0c, muc, ec, mu_rollc, mu_spinc, mc = split_params(theta, N)
        traj_cur = simulate_bullet_once_multi(args_base, v0c, muc, ec, mu_rollc, mu_spinc, mc, seconds=T_fast)
        cur_loss = traj_loss(traj_cur, gt_traj, stride=stride, weights=Wk,
                             reduction="sum", p="l2",
                             time_gamma=(1.5 if k >= K_mass else None))
        if cur_loss < best_loss:
            best_loss  = float(cur_loss)
            best_theta = theta.copy()

    # -------- 루프 종료: 'best_theta'로 최종 궤적/손실 산출 ----------
    v0b, mub, eb, mu_rollb, mu_spinb, mb = split_params(best_theta, N)
    traj_best = simulate_bullet_once_multi(args_base, v0b, mub, eb, mu_rollb, mu_spinb, mb, seconds=T_fast)

    Afin = phase_alphas(iters - 1)
    final_loss = Afin["alpha_pos"] * traj_loss(traj_best, gt_traj, stride=stride, weights=weights_all,
                                               reduction="sum", p="l2", time_gamma=1.5)
    # if k_star != -1:
    #     final_loss += Afin["alpha_dv"]  * dvel_loss_window(traj_best, gt_traj, k_contact=k_star, dt=args_base["dt"], window=win_contact)
    #     final_loss += Afin["alpha_mom"] * momentum_normal_loss(traj_best, gt_traj, k_contact=k_star, dt=args_base["dt"], masses=mb, window=win_contact)
    #     if Afin["alpha_e"] > 0.0:
    #         final_loss += Afin["alpha_e"] * e_consistency_loss(traj_best, gt_traj, k_contact=k_star, dt=args_base["dt"], window=win_contact)
    #     final_loss += Afin["alpha_mratio"] * mass_consistency_loss(gt_traj, k_contact=k_star, dt=args_base["dt"], masses=mb, window=win_contact)

    mu_bb, e_bb = mub, eb
    final_loss += lam_mu*(mu_bb - mu_prior)**2 + lam_e*(e_bb - e_prior)**2 \
                  + lam_roll*(mu_rollb - mu_roll_prior)**2 + lam_spin*(mu_spinb - mu_spin_prior)**2

    return best_theta, traj_best, hist, final_loss


def draw_overlay_o3d(sv, sf, gt_traj, best_traj, hist_list, radii):
    """다물체 궤적 시각화:
    - GT: 파란 계열(바디마다 다른 톤)
    - 학습 중 샘플(hist): 노란→빨강(이터레이션 진행색), 모든 바디
    - 최종: 초록, 모든 바디
    - radii: 길이 N 리스트(각 바디 구 반경)
    """
    import numpy as _np
    scene = o3d.geometry.TriangleMesh()
    scene.vertices = o3d.utility.Vector3dVector(sv.astype(_np.float64))
    scene.triangles = o3d.utility.Vector3iVector(sf.astype(_np.int32))
    scene.compute_vertex_normals()
    scene.paint_uniform_color([0.75, 0.75, 0.75])

    geoms = [scene]

    def make_lines(pts, color):
        pts = _np.asarray(pts)
        lines = _np.stack([_np.arange(len(pts)-1), _np.arange(1,len(pts))], axis=1)
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.colors = o3d.utility.Vector3dVector(_np.tile(_np.asarray(color, _np.float64), (len(lines),1)))
        return ls

    assert gt_traj.ndim == 3 and best_traj.ndim == 3, "traj shape should be [T+1, N, 3]"
    T, N = gt_traj.shape[0], gt_traj.shape[1]

    # 1) GT trajectories (blue-ish per body)
    gt_palette = [
        _np.array([0.10, 0.35, 1.00]),
        _np.array([0.05, 0.55, 0.95]),
        _np.array([0.20, 0.40, 0.85]),
        _np.array([0.00, 0.65, 0.90]),
    ]
    for i in range(N):
        gt_pts_i = gt_traj[:, i, :]
        geoms.append(make_lines(gt_pts_i, gt_palette[i % len(gt_palette)]))

    # 2) Hist trajectories (iteration: yellow -> red)
    if len(hist_list) > 0:
        K = len(hist_list)
        for rank, (k, t) in enumerate(hist_list):
            # 0 → yellow(1,1,0), 1 → red(1,0,0)
            a = 0.0 if K == 1 else rank/(K-1)
            col = _np.array([1.0, 1.0 - a, 0.0])
            for i in range(N):
                pts_i = t[:, i, :]
                geoms.append(make_lines(pts_i, col))

    # 3) Final trajectories (green) + spheres at start
    for i in range(N):
        best_pts_i = best_traj[:, i, :]
        geoms.append(make_lines(best_pts_i, _np.array([0.0, 0.9, 0.2])))

        ri = float(radii[i] if i < len(radii) else radii[-1])
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=ri)
        sp.translate(best_pts_i[0].astype(_np.float64))
        sp.paint_uniform_color([1.0, 0.0, 0.0])  # 시작점 구(빨강)
        geoms.append(sp)

    o3d.visualization.draw_geometries(geoms)


# ─────────────────────────────────────────────────────────────
# Main (default: generate GT → fit, Bullet only)
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--objects", type=str, nargs="+", required=True)  # first is used
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--substeps", type=int, default=6)
    ap.add_argument("--scene-rotx", type=float, default=0.0)
    ap.add_argument("--gravity", type=float, nargs=3, default=[0.0, -9.81, 0.0])
    ap.add_argument("--starts", type=float, nargs="*", default=None,
                    help="시작 위치들: x1 y1 z1 x2 y2 z2 ... (N바디)")
    ap.add_argument("--v0s", type=float, nargs="*", default=None,
                    help="v0 초기추정: v01x v01y v01z v02x v02y v02z ... (N바디)")

    # GT parameters for on-the-fly generation (Bullet)
    ap.add_argument("--gt-v0s", type=float, nargs="*", default=None, help="GT v0 (x y z)")
    ap.add_argument("--gt-mus", type=float, nargs="*", default=None, help="GT mu")
    ap.add_argument("--gt-es", type=float, nargs="*", default=None, help="GT e")
    ap.add_argument("--gt-masses", type=float, nargs="*", default=None)
    ap.add_argument("--masses", type=float, nargs="*", default=None, help="initial guess masses (for fit start)")
    ap.add_argument("--gt-save", type=str, default=None, help="optional: save generated GT traj .npy")
    ap.add_argument("--gt-mu-roll", type=float, default=0.02, help="GT rolling friction")
    ap.add_argument("--gt-mu-spin", type=float, default=0.01, help="GT spinning friction")

    # Fitting hyper-params (2-stage by default)
    ap.add_argument("--fit-2stage", action="store_true", default=True)
    ap.add_argument("--fit-iters1", type=int, default=6)
    ap.add_argument("--fit-iters2", type=int, default=6)
    ap.add_argument("--fit-fast-seconds1", type=float, default=0.5)
    ap.add_argument("--fit-fast-seconds2", type=float, default=None, help="None → full seconds")
    ap.add_argument("--fit-stride1", type=int, default=3)
    ap.add_argument("--fit-stride2", type=int, default=2)
    ap.add_argument("--fit-vmax", type=float, default=12.0, help="|vx,vy,vz| clamp")

    ap.add_argument("--viz-samples", type=int, default=10)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--save", type=str, default=None, help="save final predicted traj .npy")
    ap.add_argument("--fit-post-tail", type=float, default=1.0,
                    help="초기(질량 위주) 반복에서 k* 이후 이 초 동안 프레임을 길게 포함")

    ap.add_argument("--fit-mratio-iters", type=int, default=0,
                    help="Stage 1.5: 질량비+e 집중 최적화 반복 수(0이면 스킵)")
    ap.add_argument("--fit-mratio-seconds", type=float, default=None,
                    help="Stage 1.5: 짧은 시뮬 길이(초). None이면 k*±win 자동")
    ap.add_argument("--fit-contact-window", type=int, default=8,
                    help="k* 전후로 볼 프레임 수(±win)")

    args = ap.parse_args()

    assert len(args.objects) >= 1, "at least one dynamic object required"
    obj_path = args.objects[0]

    obj_paths = args.objects
    N = len(obj_paths)

    # starts: x1 y1 z1 x2 y2 z2 ...
    if args.starts is not None:
        starts = np.array(args.starts, dtype=np.float32).reshape(-1, 3)
        assert len(starts) == N, "--starts 길이는 objects 수와 같아야 합니다"
    else:
        starts = np.tile(np.array([[0.0, 1.0, 0.0]], np.float32), (N, 1))

    # (A) Generate GT by Bullet in this run (multi-body)
    # GT v0s: v01x v01y v01z v02x v02y v02z ...
    if args.gt_v0s is not None:
        gt_v0s = np.array(args.gt_v0s, dtype=np.float32).reshape(-1, 3)
        assert len(gt_v0s) == N, "--gt-v0s 길이는 objects 수와 같아야 합니다"
    else:
        gt_v0s = np.zeros((N, 3), np.float32)

    gt_mu = (float(args.gt_mus[0]) if args.gt_mus is not None else 0.3)
    gt_e = (float(args.gt_es[0]) if args.gt_es is not None else 0.3)

    # GT masses: 바디별
    if args.gt_masses is not None:
        gt_masses = np.array(args.gt_masses, dtype=np.float32).reshape(-1)
        assert len(gt_masses) == N, "--gt-masses 길이는 objects 수와 같아야 합니다"
    else:
        gt_masses = np.ones((N,), np.float32)

    gt = bullet_record_traj_multi(
        scene_path=args.scene,
        object_paths=obj_paths,
        seconds=args.seconds, dt=args.dt, substeps=args.substeps,
        scene_rotx_deg=args.scene_rotx, gravity=tuple(args.gravity),
        starts=starts, v0s=gt_v0s,
        mu=gt_mu, e=gt_e,
        mu_roll=args.gt_mu_roll, mu_spin=args.gt_mu_spin,
        masses=gt_masses
    )
    if args.gt_save:
        np.save(args.gt_save, gt)
        print(f"[GT] saved to {args.gt_save}  shape={gt.shape}")

    # (B) Build reusable Bullet world for fast fitting (multi-body)
    args_base = dict(
        seconds=args.seconds, dt=args.dt, substeps=args.substeps,
        scene=args.scene, objs=obj_paths,
        scene_rotx=args.scene_rotx, gravity=tuple(args.gravity),
        starts=[starts[i].tolist() for i in range(N)]
    )
    args_base["bullet_cache"] = build_bullet_cache(
        scene_obj=args.scene,
        object_paths=obj_paths,
        scene_rotx_deg=args.scene_rotx,
        start_positions=args_base["starts"],
        gravity=tuple(args.gravity),
        dt=args.dt, substeps=args.substeps
    )
    args_base["post_tail"] = float(args.fit_post_tail)
    args_base["contact_window"] = int(args.fit_contact_window)

    # (C) Initial theta (multi-body): v0(3N), mu, e, masses(N)
    # v0s 초기값: 사용자 지정 없으면 GT 앞 두 프레임으로 바디별 추정
    if args.v0s is not None:
        v0s_init = np.array(args.v0s, dtype=np.float32).reshape(-1, 3)
        assert len(v0s_init) == N, "--v0s 길이는 objects 수와 같아야 합니다"
    else:
        v0s_init = np.zeros((N, 3), np.float32)
        g = np.array(args_base["gravity"], np.float32)
        # GT에서 body마다 p0,p1로 추정
        for i in range(N):
            p0 = gt[0, i, :];
            p1 = gt[1, i, :]
            v0s_init[i, :] = (p1 - p0) / args.dt - 0.5 * g * args.dt

    # masses 초기값: 사용자 지정 없으면 1.0
    if args.masses is not None:
        masses_init = np.array(args.masses, dtype=np.float32).reshape(-1)
        assert len(masses_init) == N, "--masses 길이는 objects 수와 같아야 합니다"
    else:
        masses_init = np.ones((N,), np.float32)

    mu_init = 0.3
    e_init = 0.3
    mu_roll_init = 0.02
    mu_spin_init = 0.01
    # masses_init은 현재 코드 그대로

    theta0 = np.concatenate(
        [v0s_init.reshape(-1), [mu_init, e_init, mu_roll_init, mu_spin_init], masses_init],
        axis=0
    )

    v_bounds = ((-args.fit_vmax, args.fit_vmax),
                (-args.fit_vmax, args.fit_vmax),
                (-args.fit_vmax, args.fit_vmax))

    # 길이 = 3N + 4 + N  (mu,e,mu_roll,mu_spin = 4 scalars)
    upd1 = np.zeros(3 * N + 4 + N, dtype=np.float32);
    upd1[:3 * N] = 1.0  # Stage-1: v만
    upd2 = np.zeros_like(upd1)
    upd2[3 * N: 3 * N + 4] = 1.0  # Stage-2: mu,e,mu_roll,mu_spin
    upd2[3 * N + 4:] = 1.0  # + masses
    # Stage-2: 파라미터별 스텝 스케일 (충돌 많은 씬 안정화용)
    scale2 = upd2.astype(np.float32).copy()
    # [v0(3N) | mu e mu_roll mu_spin | masses(N)]
    scale2[0:3 * N] *= 0.0  # v0 동결
    scale2[3 * N:3 * N + 2] *= 0.5  # mu, e: 중간
    scale2[3 * N + 2:3 * N + 4] *= 0.0  # rolling, spinning: 더 작게
    scale2[3 * N + 4:] *= 1.5 # masses
    #upd2[3 * N + 4 + 0] = 0.0  # + (첫 물체 질량 고정)

    # (D) Fit (2-stage default)
    t0 = perf_counter()
    if args.fit_2stage:
        # Stage-1: v만 (짧게)
        theta1, traj1, hist1, loss1 = spsa_optimize(
            gt_traj=gt, args_base=args_base, theta0=theta0, N=N,
            iters=args.fit_iters1, a0=0.08, c0=0.15, Ak=3.0,
            stride=args.fit_stride1, fast_seconds=args.fit_fast_seconds1,
            store_k=max(1, args.viz_samples // 2),
            update_scale=upd1, v_bounds=v_bounds
        )
        # ---- Stage 1.5: contact 전후(±win)만 보고 질량비 + e만 맞추는 짧은 패스 ----
        theta_mid = theta1
        hist_mid = []
        if args.fit_mratio_iters > 0:
            N = len(obj_paths)

            # 업데이트 마스크: e + masses만 (v0/마찰류는 동결)
            upd_m = np.zeros(3 * N + 4 + N, dtype=np.float32)
            upd_m[3 * N + 1] = 1.0  # e
            upd_m[3 * N + 4:] = 1.0  # masses

            # 스텝 스케일: 질량은 크게, e는 작게
            scale_m = upd_m.copy()
            scale_m[3 * N + 1] *= 0.3  # e
            scale_m[3 * N + 4:] *= 1.2  # masses (신호-대-노이즈 ↑; 1.0~1.5 사이 조절 가능)
            # (옵션) 첫 물체 질량 고정 → 순수 질량비 식별: 아래 한 줄 주석 해제
            # scale_m[3 * N + 4 + 0] = 0.0

            # GT에서 첫 동적-동적 접촉 프레임 k* 검출
            k_star = find_first_pair_contact(gt, args_base["bullet_cache"]["radii"], eps=0.02)

            # contact 중심의 짧은 seconds 자동결정: k*±win을 덮도록
            win = int(args.fit_contact_window)
            if k_star != -1:
                # k* + win 프레임까지는 반드시 포함되도록
                sec_auto = max(args.dt * (k_star + win + 1), args.dt * (win + 2))
            else:
                sec_auto = args.fit_fast_seconds1  # 접촉이 없으면 S1 길이 fallback

            fs_mid = args.fit_mratio_seconds if args.fit_mratio_seconds is not None else min(sec_auto, args.seconds)

            # contact만 보도록 초기 반복의 'tail'을 win에 맞춰 축소
            args_base_mid = dict(**args_base)
            if k_star != -1:
                args_base_mid["post_tail"] = float(win * args.dt)

            theta_mid, traj_mid, hist_mid, loss_mid = spsa_optimize(
                gt_traj=gt, args_base=args_base_mid, theta0=theta1, N=N,
                iters=args.fit_mratio_iters, a0=0.06, c0=0.10, Ak=8.0,
                stride=1, fast_seconds=fs_mid,
                store_k=max(1, args.viz_samples // 3),
                update_scale=scale_m, v_bounds=v_bounds
            )
        # Stage-2: v 동결, μ/e + masses (길게)
        fs2 = args.seconds
        theta_seed = theta_mid if args.fit_mratio_iters > 0 else theta1
        theta, traj_best, hist2, final_loss = spsa_optimize(
            gt_traj=gt, args_base=args_base, theta0=theta_seed, N=N,
            iters=args.fit_iters2, a0=0.08, c0=0.08, Ak=10.0,
            stride=args.fit_stride2, fast_seconds=fs2,
            store_k=args.viz_samples,
            update_scale=scale2, v_bounds=v_bounds
        )
        hist = []
        hist.extend(hist1[:max(1, args.viz_samples // 2)])
        if args.fit_mratio_iters > 0:
            hist.extend(hist_mid[:max(1, args.viz_samples // 4)])
        hist.extend(hist2[:(args.viz_samples - len(hist))])
    else:
        theta, traj_best, hist, final_loss = spsa_optimize(
            gt_traj=gt, args_base=args_base, theta0=theta0, N=N,
            iters=(args.fit_iters1 + args.fit_iters2),
            a0=0.08, c0=0.15, Ak=3.0,
            stride=2, fast_seconds=0.5,
            store_k=args.viz_samples,
            update_scale=np.ones(3 * N + 2 + N, np.float32), v_bounds=v_bounds
        )
    t1 = perf_counter()

    from math import isfinite
    print(f"[FIT] done in {t1 - t0:.3f}s")
    v0_est, mu_est, e_est, mu_roll_est, mu_spin_est, m_est = split_params(theta, N)
    print(
        f"[FIT] mu={mu_est:.3f}, e={e_est:.3f}, mu_roll={mu_roll_est:.3f}, mu_spin={mu_spin_est:.3f}, masses={[round(float(x), 3) for x in m_est.tolist()]}")
    for i in range(N):
        vx, vy, vz = v0_est[3 * i:3 * i + 3]
        print(f"  v0[{i}]=({vx:.3f},{vy:.3f},{vz:.3f})")
    print(f"[FIT] loss={final_loss:.6f}")
    bw = bodywise_rms(traj_best, gt, stride=args.fit_stride2 if args.fit_2stage else args.fit_stride1,
                      weights=None)  # 가중치 영향 보고 싶으면 weights_all 넣을 수도 있음
    print(f"[FIT] per-body RMS: {bw.tolist()}")

    # (E) Overlay
    if not args.no_preview:
        sv, sf, _ = load_obj_vertices_faces(args.scene)
        if abs(args.scene_rotx) > 1e-6:
            R = rotx(args.scene_rotx); sv = (sv @ R.T).astype(np.float32)
        # 모든 object에 대해 반경 리스트 생성
        radii = []
        for _obj in obj_paths:
            ov, of, _ = load_obj_vertices_faces(_obj)
            _, r0 = compute_bounding_sphere(ov, center_hint="com")
            radii.append(r0 * 1.01)

        draw_overlay_o3d(sv, sf, gt, traj_best, hist, radii)

    if args.save:
        np.save(args.save, traj_best)
        print(f"[saved] {args.save} shape={traj_best.shape}")

    # disconnect Bullet cache
    p.disconnect(args_base["bullet_cache"]["client"])

if __name__ == "__main__":
    main()
