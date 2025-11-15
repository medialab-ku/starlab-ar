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

def build_bullet_cache(scene_obj, object_objs, scene_rotx_deg, start_positions, gravity, dt, substeps, radii_hint=None):
    """Create Bullet world once and reuse (multi-body)."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(gravity[0], gravity[1], gravity[2])

    h = float(dt)/float(max(1, substeps))
    p.setTimeStep(h)

    p.setPhysicsEngineParameter(
        numSolverIterations=50,
        useSplitImpulse=1,
        splitImpulsePenetrationThreshold=-0.02,
        deterministicOverlappingPairs=1,
        restitutionVelocityThreshold=0.0
    )

    # scene (static concave mesh)
    flags = p.GEOM_FORCE_CONCAVE_TRIMESH
    scene_col = p.createCollisionShape(p.GEOM_MESH, fileName=scene_obj, flags=flags)
    scene_vis = p.createVisualShape(p.GEOM_MESH, fileName=scene_obj)
    rx = math.radians(scene_rotx_deg)
    scene_quat = p.getQuaternionFromEuler([rx, 0.0, 0.0])
    scene_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=scene_col,
                                 baseVisualShapeIndex=scene_vis,
                                 basePosition=[0,0,0], baseOrientation=scene_quat)

    # dynamic spheres (bounding spheres of objects)
    body_ids = []
    radii = []
    sphere_shapes = []  # (col_id, vis_id) per body
    for i, obj_path in enumerate(object_objs):
        ov, of, _ = load_obj_vertices_faces(obj_path)
        _, r_obj = compute_bounding_sphere(ov, center_hint="com")
        r = float((radii_hint[i] if radii_hint is not None else r_obj)*1.01)

        col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[1,0,0,1])
        sphere_shapes.append((col, vis))
        start_i = start_positions[i]
        bid = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis,
                                basePosition=start_i, baseOrientation=[0,0,0,1])
        body_ids.append(bid)
        radii.append(r)

    return dict(client=cid, scene_id=scene_id, body_ids=body_ids,
                radii=radii, h=h, substeps=substeps, starts=start_positions,
                scene_quat=scene_quat, sphere_shapes=sphere_shapes)

def simulate_bullet_once_multi(args_base, v0_flat, mu, e, mu_roll, mu_spin, masses, seconds=None):

    """Generate one trajectory for N bodies using cached Bullet world.
       v0_flat: length 3N (vx1,vy1,vz1, vx2,vy2,vz2, ...)
       masses: list length N
       returns traj: [T+1, N, 3]
    """
    cache = args_base["bullet_cache"]
    body_ids = cache["body_ids"]
    N = len(body_ids)
    assert len(v0_flat) == 3*N
    assert len(masses) == N

    # === recreate dynamic bodies with desired masses ===
    starts = cache["starts"]
    substeps = cache["substeps"]
    new_body_ids = []
    # 기존 바디 제거
    for bid in body_ids:
        p.removeBody(bid)
    # 동일 shape로 새로 생성 (mass 반영)
    for i in range(N):
        vx, vy, vz = v0_flat[3 * i:3 * i + 3]
        col_id, vis_id = cache["sphere_shapes"][i]
        bid = p.createMultiBody(
            baseMass=float(masses[i]),
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=starts[i],
            baseOrientation=[0, 0, 0, 1]
        )
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
        p.resetBaseVelocity(bid, [float(vx), float(vy), float(vz)], [0, 0, 0])
        new_body_ids.append(bid)
    cache["body_ids"] = new_body_ids
    body_ids = new_body_ids

    seconds = args_base["seconds"] if seconds is None else seconds
    steps = int(seconds/args_base["dt"])
    substeps = cache["substeps"]
    traj = np.zeros((steps+1, N, 3), dtype=np.float32)

    # initial samples
    for i, bid in enumerate(body_ids):
        pos, _ = p.getBasePositionAndOrientation(bid)
        traj[0, i, :] = np.array(pos, dtype=np.float32)

    for s in range(1, steps+1):
        for _ in range(substeps):
            p.stepSimulation()
        for i, bid in enumerate(body_ids):
            pos, _ = p.getBasePositionAndOrientation(bid)
            traj[s, i, :] = np.array(pos, dtype=np.float32)

    return traj

def bullet_record_traj_multi(scene_path, object_paths, seconds, dt, substeps,
                             scene_rotx_deg, gravity, starts, v0s, mu, e, mu_roll, mu_spin, masses):
    """One-shot world build for GT; returns [T+1, N, 3] and disconnects."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(*gravity)

    h = float(dt)/max(1, int(substeps))
    p.setTimeStep(h)

    p.setPhysicsEngineParameter(
        numSolverIterations=50,
        useSplitImpulse=1,
        splitImpulsePenetrationThreshold=-0.02,
        deterministicOverlappingPairs=1,
        restitutionVelocityThreshold=0.0
    )

    flags = p.GEOM_FORCE_CONCAVE_TRIMESH
    scene_col = p.createCollisionShape(p.GEOM_MESH, fileName=scene_path, flags=flags)
    scene_vis = p.createVisualShape(p.GEOM_MESH, fileName=scene_path)
    rx = math.radians(scene_rotx_deg)
    scene_quat = p.getQuaternionFromEuler([rx, 0.0, 0.0])
    _scene_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=scene_col,
                                  baseVisualShapeIndex=scene_vis,
                                  basePosition=[0,0,0], baseOrientation=scene_quat)

    body_ids = []
    radii = []
    for i, obj in enumerate(object_paths):
        ov, of, _ = load_obj_vertices_faces(obj)
        _, r_obj = compute_bounding_sphere(ov, center_hint="com")
        r = float(r_obj*1.01)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[1,0,0,1])
        bid = p.createMultiBody(baseMass=float(masses[i]),
                                baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis,
                                basePosition=list(starts[i]),
                                baseOrientation=[0,0,0,1])
        p.changeDynamics(
            bid, -1,
            lateralFriction=float(mu),
            restitution=float(e),
            rollingFriction=float(mu_roll),
            spinningFriction=float(mu_spin),
            frictionAnchor=1,
            linearDamping=0.0,
            angularDamping=0.0
        )
        p.resetBaseVelocity(bid, list(v0s[i]), [0,0,0])
        body_ids.append(bid); radii.append(r)

    steps = int(seconds/dt)
    N = len(body_ids)
    traj = np.zeros((steps+1, N, 3), dtype=np.float32)
    for i,bid in enumerate(body_ids):
        pos,_ = p.getBasePositionAndOrientation(bid)
        traj[0,i,:] = np.asarray(pos, np.float32)

    for s in range(1, steps+1):
        for _ in range(substeps): p.stepSimulation()
        for i,bid in enumerate(body_ids):
            pos,_ = p.getBasePositionAndOrientation(bid)
            traj[s,i,:] = np.asarray(pos, np.float32)

    p.disconnect(cid)
    return traj
# ─────────────────────────────────────────────────────────────
# Loss / Opt / Viz
# ─────────────────────────────────────────────────────────────

def traj_loss(traj_pred, traj_gt, stride=1, weights=None,
              reduction="sum",    # "mean" | "sum"
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

    pure_loss = float(np.sum(per_frame))

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
        return float(np.sum(per_frame)), pure_loss
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

    #time count 1: 초반 파라미터 설정 및 보정
    t0 = perf_counter()
    duration = 0
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

    t1 = perf_counter()
    duration = t1 - t0
    # time count 1 종료

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
    best_loss  = (float("inf"), float("inf"))
    length = gt_traj.shape[0]
    for k in range(iters):

        #time count 2: 파라미터 할당 및 계산
        t0 = perf_counter()
        ck = c0 / ((k + 1.0) ** gamma)
        ak = a0 / ((k + 1.0 + Ak) ** alpha)
        delta = np.array([1 if rng.random() < 0.5 else -1 for _ in range(len(theta))], dtype=np.float32)

        thetap = clamp_params(theta + ck * delta, N, v_bounds=v_bounds)
        thetam = clamp_params(theta - ck * delta, N, v_bounds=v_bounds)

        v0p, mup, ep, mu_rollp, mu_spinp, mp = split_params(thetap, N)
        v0m, mum, em, mu_rollm, mu_spinm, mm = split_params(thetam, N)
        t1 = perf_counter()
        #time count 2 종료

        traj_p = simulate_bullet_once_multi(args_base, v0p, mup, ep, mu_rollp, mu_spinp, mp, seconds=T_fast)
        traj_m = simulate_bullet_once_multi(args_base, v0m, mum, em, mu_rollm, mu_spinm, mm, seconds=T_fast)

        #time count 3: spsa 업데이트 계산
        t2 = perf_counter()

        A  = phase_alphas(k)
        Wk = phase_weights(k, weights_all, k_star, T_total=len(weights_all))

        # 기본 위치/시간가중 손실 (sum: 후반 누적오차 민감)
        fp = A["alpha_pos"] * traj_loss(traj_p, gt_traj, stride=stride, weights=Wk,
                                        reduction="sum", p="l2",
                                        time_gamma=(1.5 if k >= K_mass else None))[0]
        fm = A["alpha_pos"] * traj_loss(traj_m, gt_traj, stride=stride, weights=Wk,
                                        reduction="sum", p="l2",
                                        time_gamma=(1.5 if k >= K_mass else None))[0]

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
        t3 = perf_counter()
        #time count 3 종료

        # ---------- 변경점 ①: 히스토리 저장 (마지막 반복은 반드시 저장) ----------
        if (k + 1) == iters or len(hist) < store_k:
            v0c, muc, ec, mu_rollc, mu_spinc, mc = split_params(theta, N)
            traj_now = simulate_bullet_once_multi(args_base, v0c, muc, ec, mu_rollc, mu_spinc, mc, seconds=T_fast)
            hist.append((k, traj_now))

        #time count 4: best loss 갱신
        t4 = perf_counter()
        # ---------- 변경점 ②: '현재 θ' 검증손실로 best 갱신 ----------
        # 같은 창/가중으로 간단 검증
        v0c, muc, ec, mu_rollc, mu_spinc, mc = split_params(theta, N)
        traj_cur = simulate_bullet_once_multi(args_base, v0c, muc, ec, mu_rollc, mu_spinc, mc, seconds=T_fast)
        cur_loss = traj_loss(traj_cur, gt_traj, stride=stride, weights=Wk,
                             reduction="sum", p="l2",
                             time_gamma=(1.5 if k >= K_mass else None))
        print(f"[FIT] Loss for iteration {k + 1} is {cur_loss[1]/length}")
        if cur_loss[0] < best_loss[0]:
            best_loss  = cur_loss
            best_theta = theta.copy()
        t5 = perf_counter()
        #time count 4 종료 및 잰 시간 합산
        duration += t5-t4 + t3-t2 + t1-t0

    #time count 5: 최종 궤적 산출
    t0 = perf_counter()
    # -------- 루프 종료: 'best_theta'로 최종 궤적/손실 산출 ----------
    v0b, mub, eb, mu_rollb, mu_spinb, mb = split_params(best_theta, N)
    traj_best = simulate_bullet_once_multi(args_base, v0b, mub, eb, mu_rollb, mu_spinb, mb, seconds=T_fast)

    Afin = phase_alphas(iters - 1)
    final_loss = Afin["alpha_pos"] * traj_loss(traj_best, gt_traj, stride=stride, weights=weights_all,
                                               reduction="sum", p="l2", time_gamma=1.5)[0]

    mu_bb, e_bb = mub, eb
    final_loss += lam_mu*(mu_bb - mu_prior)**2 + lam_e*(e_bb - e_prior)**2 \
                  + lam_roll*(mu_rollb - mu_roll_prior)**2 + lam_spin*(mu_spinb - mu_spin_prior)**2
    t1 = perf_counter()
    duration += t1-t0
    #time count 5 종료

    return best_theta, traj_best, hist, best_loss, duration

# ─────────────────────────────────────────────────────────────
# Main (default: generate GT → fit, Bullet only)
# ─────────────────────────────────────────────────────────────

def simulate(scene, objects, seconds, dt, substeps, rotation, gravity, starts, gt_traj, viz_samples=10,
             fit_iters1=2, fit_iters2=8, fit_fast_seconds1=0.1, fit_stride1=1, fit_stride2=2, fit_vmax=12, fit_post_tail=1):

    assert len(objects) >= 1, "at least one dynamic object required"
    obj_paths = objects
    N = len(obj_paths)

    # starts: x1 y1 z1 x2 y2 z2 ...
    if starts is not None:
        starts = np.array(starts, dtype=np.float32).reshape(-1, 3)
        assert len(starts) == N, "--starts 길이는 objects 수와 같아야 합니다"
    else:
        starts = np.tile(np.array([[0.0, 1.0, 0.0]], np.float32), (N, 1))

    # (A) Generate GT by Bullet in this run (multi-body)
    # GT v0s: v01x v01y v01z v02x v02y v02z ...
    #shape: nparray, dtype float32, reshape(-1,3)
    #mu, e: float
    #masses: nparray, float32, reshape(-1)

    # Build reusable Bullet world for fast fitting (multi-body)
    args_base = dict(
        seconds=seconds, dt=dt, substeps=substeps,
        scene=scene, objs=obj_paths,
        scene_rotx=rotation, gravity=gravity,
        starts=[starts[i].tolist() for i in range(N)]
    )
    args_base["bullet_cache"] = build_bullet_cache(
        scene_obj=scene,
        object_objs=obj_paths,
        scene_rotx_deg=rotation,
        start_positions=args_base["starts"],
        gravity=gravity,
        dt=dt, substeps=substeps
    )
    args_base["post_tail"] = float(fit_post_tail)
    # (C) Initial theta (multi-body): v0(3N), mu, e, masses(N)
    # v0s 초기값: 사용자 지정 없으면 GT 앞 두 프레임으로 바디별 추정
    v0s_init = np.zeros((N, 3), np.float32)
    g = np.array(args_base["gravity"], np.float32)
    # GT에서 body마다 p0,p1로 추정
    for i in range(N):
        p0 = gt_traj[0, i, :]
        p1 = gt_traj[1, i, :]
        v0s_init[i, :] = (p1 - p0) / dt - 0.5 * g * dt

    # masses 초기값: 사용자 지정 없으면 1.0
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

    v_bounds = ((-fit_vmax, fit_vmax),
                (-fit_vmax, fit_vmax),
                (-fit_vmax, fit_vmax))

    # 길이 = 3N + 4 + N  (mu,e,mu_roll,mu_spin = 4 scalars)
    upd1 = np.zeros(3 * N + 4 + N, dtype=np.float32);
    upd1[:3 * N] = 1.0  # Stage-1: v만
    upd2 = np.zeros_like(upd1)
    upd2[3 * N: 3 * N + 4] = 1.0  # Stage-2: mu,e,mu_roll,mu_spin
    upd2[3 * N + 4:] = 1.0  # + masses
    # Stage-2: 파라미터별 스텝 스케일 (충돌 많은 씬 안정화용)
    scale2 = upd2.astype(np.float32).copy()
    scale2[0:3 * N] *= 0.0  # v0 동결
    scale2[3 * N:3 * N + 2] *= 0.5  # mu, e: 중간
    scale2[3 * N + 2:3 * N + 4] *= 0.0  # rolling, spinning: 더 작게
    scale2[3 * N + 4:] *= 1.5 # masses
    print("init values calculated")
    # (D) Fit (2-stage default)
    # Stage-1: v만 (짧게)
    theta1, traj1, hist1, loss1, duration1 = spsa_optimize(
        gt_traj=gt_traj, args_base=args_base, theta0=theta0, N=N,
        iters=fit_iters1, a0=0.08, c0=0.15, Ak=3.0,
        stride=fit_stride1, fast_seconds=fit_fast_seconds1,
        store_k=max(1, viz_samples // 2),
        update_scale=upd1, v_bounds=v_bounds
    )
    # Stage-2: v 동결, μ/e + masses (길게)
    theta_seed = theta1
    theta, traj_best, hist2, final_loss, duration2 = spsa_optimize(
        gt_traj=gt_traj, args_base=args_base, theta0=theta_seed, N=N,
        iters=fit_iters2, a0=0.08, c0=0.08, Ak=10.0,
        stride=fit_stride2, fast_seconds=seconds,
        store_k=viz_samples,
        update_scale=scale2, v_bounds=v_bounds
    )
    hist = []
    hist.extend(hist1[:max(1, viz_samples // 2)])
    hist.extend(hist2[:(viz_samples - len(hist))])
    duration = duration1 + duration2
    from math import isfinite
    print(f"[FIT] done in {duration:.3f}s")
    v0_est, mu_est, e_est, mu_roll_est, mu_spin_est, m_est = split_params(theta, N)
    print(
        f"[FIT] mu={mu_est:.3f}, e={e_est:.3f}, mu_roll={mu_roll_est:.3f}, mu_spin={mu_spin_est:.3f}, masses={[round(float(x), 3) for x in m_est.tolist()]}")
    for i in range(N):
        vx, vy, vz = v0_est[3 * i:3 * i + 3]
        print(f"  v0[{i}]=({vx:.3f},{vy:.3f},{vz:.3f})")
    length = gt_traj.shape[0]
    print(f"[FIT] loss={final_loss[1]/length:.6f}")
    bw = bodywise_rms(traj_best, gt_traj, stride=fit_stride2 ,weights=None)  # 가중치 영향 보고 싶으면 weights_all 넣을 수도 있음
    print(f"[FIT] per-body RMS: {bw.tolist()}")

    # disconnect Bullet cache
    p.disconnect(args_base["bullet_cache"]["client"])

    return theta, traj_best, hist, final_loss, duration

if __name__ == "__main__":
    main()
