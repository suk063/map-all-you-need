"""Component-local DINO maps rendered with MuJoCo, updated with MJX poses."""

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import trimesh
from mujoco.mjx._src.math import quat_to_mat
from mujoco_playground._src.wrapper import Wrapper

from benchmark.common.envs import GOAL_BODIES, GOAL_TASKS

MAP_VERSION = 2
IMAGE_SIZE = 256
# Explicit robot roots keep static scene furniture out of the observation.
ROBOT_ROOTS = {
    "Panda": ("link0",),
    "Aloha": ("left/base_link", "right/base_link"),
    "Leap": ("leap_mount",),
    "Aero": ("tetheria_mount",),
}
GRIPPER_ROOTS = {
    "Panda": ("hand",),
    "Aloha": ("left/gripper_link", "right/gripper_link"),
    "Leap": ("palm",),
    "Aero": ("palm",),
}
# Native manipulated bodies, including every child part of an articulation.
TASK_OBJECTS = {
    "AlohaHandOver": ("box",),
    "AlohaSinglePegInsertion": ("peg", "socket"),
    "PandaPickCube": ("box",),
    "PandaPickCubeOrientation": ("box",),
    "PandaPickCubeCartesian": ("box",),
    "PandaOpenCabinet": ("handle",),
    "PandaRobotiqPushCube": ("box",),
    "LeapCubeReorient": ("cube",),
    "LeapCubeRotateZAxis": ("cube",),
    "AeroCubeRotateZAxis": ("cube",),
}


def descendants(model, roots):
    ids = {model.body(name).id for name in roots}
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in ids:
            ids.add(body)
    return ids


def components(model, env_id, config):
    if type(config["background"]) is not bool:
        raise ValueError("Map background must be true or false; legacy table/none configs are unsupported")
    robot = config.get("robot", "full")
    if robot not in ("full", "gripper"):
        raise ValueError("Map robot must be full or gripper")
    family = next(key for key in ROBOT_ROOTS if env_id.startswith(key))
    all_robot = descendants(model, ROBOT_ROOTS[family])
    roots = ("base",) if env_id == "PandaRobotiqPushCube" else GRIPPER_ROOTS[family]
    selected_robot = all_robot if robot == "full" else descendants(model, roots)
    goals = descendants(model, [name for name in GOAL_BODIES if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0])
    required = selected_robot | descendants(model, TASK_OBJECTS[env_id]) | goals
    groups = {}
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        # Background selection must not bring excluded arm links or mounts back.
        if body in all_robot and body not in selected_robot:
            continue
        if body in goals and env_id not in GOAL_TASKS:
            continue
        material = int(model.geom_matid[geom])
        alpha = model.mat_rgba[material, 3] if material >= 0 else model.geom_rgba[geom, 3]
        # Standard MuJoCo visual groups are 0, 1, 2; group 3 holds collision proxies.
        if model.geom_group[geom] > 2 or alpha <= 0 or model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        body_name = model.body(body).name
        mesh_id = int(model.geom_dataid[geom])
        mesh_name = model.mesh(mesh_id).name if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH else ""
        label = (model.geom(geom).name + " " + body_name + " " + mesh_name).lower()
        if any(word in label for word in ("floor", "ground", "wall", "barrier")):
            continue
        tabletop = "tabletop" in label or label.strip() == "table"
        # Aloha's tabletop is used by its no_table_collision reward.
        task_surface = family == "Aloha" and tabletop
        if body not in required and not task_surface and not config["background"]:
            continue
        # World geoms can be separate furniture pieces, not one giant object.
        groups.setdefault((body, geom if body == 0 else -1, tabletop), []).append(geom)
    if not groups:
        raise ValueError(f"No map components selected for {env_id}")
    return [(body, geoms, table) for (body, _, table), geoms in groups.items()]


def quat_matrix(quat):
    out = np.empty(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=np.float64))
    return out.reshape(3, 3)


def visual_mesh(model, geoms):
    meshes = []
    for geom in geoms:
        kind, size = mujoco.mjtGeom(int(model.geom_type[geom])), model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[geom]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            mesh = trimesh.Trimesh(model.mesh_vert[va:va + vn].copy(), model.mesh_face[fa:fa + fn].copy(), process=False)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2 * size)
        elif kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            mesh = trimesh.creation.icosphere(subdivisions=3)
            mesh.apply_scale(size[0] if kind == mujoco.mjtGeom.mjGEOM_SPHERE else size)
        elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            factory = trimesh.creation.capsule if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else trimesh.creation.cylinder
            mesh = factory(radius=size[0], height=2 * size[1])
        else:
            raise ValueError(f"Unsupported map geom type {kind}: {model.geom(geom).name}")
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = quat_matrix(model.geom_quat[geom]), model.geom_pos[geom]
        mesh.apply_transform(transform)
        meshes.append(mesh)
    return trimesh.util.concatenate(meshes)


def surface_points(mesh, spacing, tabletop=False):
    retained = {}
    top = mesh.bounds[1, 2]
    for triangle, normal in zip(mesh.triangles, mesh.face_normals):
        if not np.isfinite(normal).all() or np.linalg.norm(normal) < .5:
            continue
        if tabletop and (normal[2] < .9 or triangle[:, 2].max() < top - .005):
            continue
        divisions = max(1, int(np.ceil(np.linalg.norm(np.roll(triangle, -1, axis=0) - triangle, axis=1).max() / spacing)))
        for i in range(divisions + 1):
            j = np.arange(divisions - i + 1) / divisions
            points = triangle[0] + (i / divisions) * (triangle[1] - triangle[0]) + j[:, None] * (triangle[2] - triangle[0])
            for point in points:
                retained.setdefault(tuple(np.floor(point / spacing).astype(np.int64)), (point, normal))
    if not retained:
        raise ValueError("No surface points in selected visual mesh")
    ordered = [retained[key] for key in sorted(retained)]
    return np.stack([p for p, _ in ordered]), np.stack([n for _, n in ordered])


def sphere_views(count):
    k = np.arange(count)
    z = 1 - 2 * (k + .5) / count
    angle = k * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z * z)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def appearance_key(model, geoms, mesh, config, tabletop):
    digest = hashlib.sha256()
    settings = {key: config[key] for key in ("voxel_size", "views", "extra_views")}
    settings.update(version=MAP_VERSION, mujoco=mujoco.__version__, tabletop=tabletop,
                    dino_sha256=config["dino"]["sha256"], source_revision=config["dino"]["source_revision"],
                    image_size=IMAGE_SIZE, lighting="headlight_0.5_0.8_0", render_tendons=False)
    digest.update(json.dumps(settings, sort_keys=True).encode())
    for value in (mesh.vertices, mesh.faces, model.geom_rgba[geoms], model.geom_size[geoms],
                  model.geom_pos[geoms], model.geom_quat[geoms], model.geom_type[geoms]):
        digest.update(np.asarray(value).tobytes())
    for geom in geoms:
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom]
            for field in ('normal', 'texcoord'):
                start, count = getattr(model, f'mesh_{field}adr')[mesh_id], getattr(model, f'mesh_{field}num')[mesh_id]
                if start >= 0:
                    digest.update(getattr(model, f'mesh_{field}')[start:start + count].tobytes())
            start, count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
            for field in ('mesh_facenormal', 'mesh_facetexcoord'):
                digest.update(getattr(model, field)[start:start + count].tobytes())
        mid = model.geom_matid[geom]
        if mid < 0:
            continue
        for name in ("mat_rgba", "mat_texrepeat", "mat_texuniform", "mat_emission", "mat_specular", "mat_shininess", "mat_reflectance"):
            digest.update(np.asarray(getattr(model, name)[mid]).tobytes())
        for role, tid in enumerate(np.asarray(model.mat_texid[mid]).ravel()):
            if tid >= 0:
                digest.update(np.array([role, model.tex_type[tid], model.tex_width[tid],
                                        model.tex_height[tid], model.tex_nchannel[tid]], dtype=np.int64).tobytes())
                start = model.tex_adr[tid]
                count = model.tex_width[tid] * model.tex_height[tid] * model.tex_nchannel[tid]
                digest.update(model.tex_data[start:start + count].tobytes())
    return digest.hexdigest()


class IsolatedRenderer:
    """A private render model; only the selected component's geoms are visible."""

    def __init__(self, model, geoms):
        self.model = copy.copy(model)
        self.model.geom_group[:] = 5
        self.model.geom_group[geoms] = 2
        self.model.stat.extent = 1
        self.model.vis.map.znear, self.model.vis.map.zfar = .001, 100
        self.model.vis.global_.offwidth = self.model.vis.global_.offheight = IMAGE_SIZE
        self.model.vis.global_.fovy = 60
        # Fixed lighting makes a component cache independent of its task scene.
        self.model.light_active[:] = 0
        self.model.vis.headlight.active = 1
        self.model.vis.headlight.ambient[:] = .5
        self.model.vis.headlight.diffuse[:] = .8
        self.model.vis.headlight.specular[:] = 0
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        # Render in this body's local frame, preserving compiled mesh transforms.
        self.data.geom_xpos[geoms] = model.geom_pos[geoms]
        self.data.geom_xmat[geoms] = np.stack([quat_matrix(model.geom_quat[g]).ravel() for g in geoms])
        self.renderer = mujoco.Renderer(self.model, height=IMAGE_SIZE, width=IMAGE_SIZE)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[2] = 1
        self.option.sitegroup[:] = 0
        self.option.tendongroup[:] = 0
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    def render(self, eye, target):
        direction = np.asarray(eye) - target
        self.camera.lookat[:] = target
        self.camera.distance = np.linalg.norm(direction)
        self.camera.azimuth = 180 + np.degrees(np.arctan2(direction[1], direction[0]))
        self.camera.elevation = -np.degrees(np.arcsin(direction[2] / self.camera.distance))
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        # Average the stereo cameras to obtain the actual mono render camera.
        camera = mujoco.mjv_averageCamera(*self.renderer.scene.camera)
        forward, up = np.array(camera.forward), np.array(camera.up)
        right = np.cross(forward, up)
        rotation = np.column_stack((right, up, -forward))
        focal = IMAGE_SIZE * camera.frustum_near / (camera.frustum_top - camera.frustum_bottom)
        return rgb, depth, np.array(camera.pos), rotation, focal

    def close(self):
        self.renderer.close()


def observe(renderer, net, xyz, eye, target, device):
    import torch
    from torch.nn import functional as F

    rgb, depth, position, rotation, focal = renderer.render(eye, target)
    camera = (xyz - position) @ rotation
    z = -camera[:, 2]
    uv = np.column_stack((camera[:, 0], -camera[:, 1])) / np.maximum(z[:, None], 1e-9)
    uv = uv * focal + (IMAGE_SIZE - 1) / 2
    pixels = np.rint(uv).astype(int)
    valid = (z > .001) & (pixels >= 0).all(1) & (pixels < IMAGE_SIZE).all(1)
    ids = np.flatnonzero(valid)
    ids = ids[np.abs(depth[pixels[ids, 1], pixels[ids, 0]] - z[ids]) < np.maximum(.0015, .002 * z[ids])]
    if not len(ids):
        return ids, np.empty((0, 1024), np.float32)
    with torch.no_grad():
        tokens = net(torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device))
        features = tokens.transpose(1, 2).reshape(1, 1024, 16, 16)
        grid = torch.tensor(2 * (uv[ids] + .5) / IMAGE_SIZE - 1, device=device, dtype=torch.float32)[None, :, None]
        samples = F.grid_sample(features.float(), grid, align_corners=False, padding_mode="border")
    return ids, samples[0, :, :, 0].T.cpu().numpy()


def build_template(model, geoms, mesh, config, destination, net, device, tabletop=False):
    xyz, normals = surface_points(mesh, config["voxel_size"], tabletop)
    sums, counts = np.zeros((len(xyz), 1024), np.float32), np.zeros(len(xyz), np.int32)
    renderer = IsolatedRenderer(model, geoms)
    start, used = time.monotonic(), 0
    try:
        center = (xyz.min(0) + xyz.max(0)) / 2
        radius = max(np.linalg.norm(xyz - center, axis=1).max() * 2.5, .05)
        for direction in sphere_views(config["views"]):
            ids, features = observe(renderer, net, xyz, center + radius * direction, center, device)
            sums[ids] += features
            counts[ids] += 1
            used += 1
        for extra in range(config["extra_views"]):
            unseen = np.flatnonzero(counts == 0)
            if not len(unseen):
                break
            index = unseen[extra % len(unseen)]
            eye = xyz[index] + normals[index] * (.025, .06, .15)[extra % 3]
            ids, features = observe(renderer, net, xyz, eye, xyz[index], device)
            sums[ids] += features
            counts[ids] += 1
            used += 1
    finally:
        renderer.close()
    keep = counts > 0
    if not keep.any():
        raise ValueError(f"No observed surface in {destination}; inspect map rendering")
    report = {"candidate_points": len(xyz), "points": int(keep.sum()), "coverage": float(keep.mean()),
              "views": used, "seconds": time.monotonic() - start, "config": config}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    with h5py.File(temporary, "w") as f:
        f.attrs.update(format_version=MAP_VERSION, metadata=json.dumps(report))
        f.create_dataset("xyz", data=xyz[keep].astype(np.float32))
        f.create_dataset("features", data=(sums[keep] / counts[keep, None]).astype(np.float16), compression="gzip")
        f.create_dataset("counts", data=counts[keep])
    temporary.replace(destination)
    print(f"Map {destination.stem[:12]}: {keep.sum()}/{len(xyz)} points, {used} views, {report['seconds']:.1f}s", flush=True)


class FeatureBank:
    def __init__(self):
        self.paths, self.templates, self.arrays = [], {}, []
        self.size = 0

    def add(self, path):
        path = str(Path(path).resolve())
        if path not in self.templates:
            with h5py.File(path) as f:
                if f.attrs.get("format_version") != MAP_VERSION:
                    raise ValueError(f"Incompatible map cache: {path}")
                xyz, features = f["xyz"][:], f["features"][:]
            if xyz.shape != (len(xyz), 3) or not len(xyz) or features.shape != (len(xyz), 1024):
                raise ValueError(f"Invalid map cache shapes: {path}")
            if not np.isfinite(features).all() or not np.isfinite(xyz).all():
                raise ValueError(f"Nonfinite map cache: {path}")
            ids = np.arange(self.size, self.size + len(xyz), dtype=np.int32)
            self.templates[path] = (xyz, ids)
            self.arrays.append(features)
            self.paths.append(path)
            self.size += len(xyz)
        return self.templates[path]

    @property
    def features(self):
        return jnp.asarray(np.concatenate(self.arrays), dtype=jnp.float32)


class MapObservationWrapper(Wrapper):
    def __init__(self, env, env_id, config):
        super().__init__(env)
        self.bank = FeatureBank()
        local, ids, bodies = [], [], []
        self.component_names = []
        net = None
        try:
            for body, geoms, table in components(env.mj_model, env_id, config):
                mesh = visual_mesh(env.mj_model, geoms)
                key = appearance_key(env.mj_model, geoms, mesh, config, table)
                path = Path(config["cache"]) / f"{key}.h5"
                if not path.is_file():
                    import torch

                    from benchmark.common.dino import FrozenDINO
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    if net is None:
                        with torch.random.fork_rng():
                            net = FrozenDINO(config["dino"]).to(device)
                    build_template(env.mj_model, geoms, mesh, config, path, net, device, table)
                xyz, features = self.bank.add(path)
                local.append(xyz)
                ids.append(features)
                bodies.extend([body] * len(xyz))
                name = env.mj_model.body(body).name
                if body == 0:
                    geom = geoms[0]
                    name = f"world/{env.mj_model.geom(geom).name or 'geom'}:{geom}"
                self.component_names.append(name)
        finally:
            if net is not None:
                del net
                torch.cuda.empty_cache()
        self.local = jnp.asarray(np.concatenate(local))
        self.feature_ids = jnp.asarray(np.concatenate(ids))
        self.body_ids = jnp.asarray(bodies)
        mocap_ids = env.mj_model.body_mocapid[np.asarray(bodies)]
        self.mocap_points = jnp.asarray(np.flatnonzero(mocap_ids >= 0))
        self.mocap_ids = jnp.asarray(mocap_ids[mocap_ids >= 0])

    @property
    def observation_size(self):
        n = len(self.feature_ids)
        return {"xyz": (n * 3,), "feature_ids": (n,)}

    def observation(self, data):
        rotation = data.xmat[self.body_ids]
        position = data.xpos[self.body_ids]
        if self.mocap_points.size:
            # Native tasks can update a goal after physics; xpos/xmat then lag
            # behind mocap. Read its current pose without advancing the simulator.
            rotation = rotation.at[self.mocap_points].set(jax.vmap(quat_to_mat)(data.mocap_quat[self.mocap_ids]))
            position = position.at[self.mocap_points].set(data.mocap_pos[self.mocap_ids])
        xyz = jnp.einsum("nij,nj->ni", rotation, self.local) + position
        # Flat leaves keep Brax's rollout/statistics batch axes consistent.
        return {"xyz": xyz.reshape(-1), "feature_ids": self.feature_ids}

    def _replace_obs(self, state):
        return state.replace(obs=self.observation(state.data), info={**state.info, "_map_native_obs": state.obs})

    def reset(self, rng):
        state = self.env.reset(rng)
        state = state.replace(info={**state.info, "_map_first_obs": state.obs, "_map_reset_count": jnp.array(0)})
        return self._replace_obs(state)

    def step(self, state, action):
        # Match upstream's cached-observation autoreset, including history observations.
        count = state.info.get("AutoResetWrapper_done_count", state.info["_map_reset_count"])
        reset = count != state.info["_map_reset_count"]
        obs = jax.tree.map(lambda first, last: jnp.where(reset, first, last),
                           state.info["_map_first_obs"], state.info["_map_native_obs"])
        native = state.replace(obs=obs, info={**state.info, "_map_reset_count": count})
        return self._replace_obs(self.env.step(native, action))
