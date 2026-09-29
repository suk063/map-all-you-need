"""Cached component-local DINO maps, assembled from simulator poses before auto-reset."""

import hashlib
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from torch.nn import functional as F

from benchmark.common.dino import FrozenDINO, file_hash

MAP_VERSION = 1
GRIPPERS = {
    "panda": {"panda_hand", "panda_leftfinger", "panda_rightfinger"},
    "panda_wristcam": {"panda_hand", "panda_leftfinger", "panda_rightfinger"},
    "panda_stick": {"panda_hand"},
    "so100": {"Fixed_Jaw", "Moving_Jaw"},
    "widowxai": {"link_6", "carriage_left", "carriage_right", "gripper_left", "gripper_right"},
}


def batch_observations(observations):
    """Concatenate batches, padding variable-size point maps (ID -1 means padding)."""
    if "xyz" not in observations[0]:
        return {k: torch.cat([o[k] for o in observations]) for k in observations[0]}
    size = max(o["xyz"].shape[1] for o in observations)
    return {
        "xyz": torch.cat([F.pad(o["xyz"], (0, 0, 0, size - o["xyz"].shape[1])) for o in observations]),
        "feature_ids": torch.cat([F.pad(o["feature_ids"], (0, size - o["feature_ids"].shape[1]), value=-1) for o in observations]),
    }


class FeatureBank:
    """One immutable feature row per cached point; rollout snapshots hold only row IDs."""
    def __init__(self, paths=()):
        self.paths, self.templates, self.arrays = [], {}, []
        self.devices, self.size = {}, 0
        for path in paths:
            self.add(path)

    def add(self, path):
        path = str(Path(path).resolve())
        if path not in self.templates:
            with h5py.File(path) as f:
                if f.attrs.get("format_version") != MAP_VERSION:
                    raise ValueError(f"Incompatible map cache: {path}")
                xyz, features = f["xyz"][:], f["features"][:]
                if len(xyz) == 0 or features.shape != (len(xyz), 1024) or not np.isfinite(features).all() or not np.isfinite(xyz).all():
                    raise ValueError(f"Empty/nonfinite map cache: {path}")
            self.templates[path] = (torch.from_numpy(xyz), torch.arange(self.size, self.size + len(xyz)))
            self.arrays.append(torch.from_numpy(features))
            self.paths.append(path)
            self.size += len(xyz)
            self.devices.clear()
        return self.templates[path]

    def lookup(self, ids):
        if not self.arrays:
            raise ValueError("No map features loaded; create the environment with the policy's map bank")
        key = str(ids.device)
        if key not in self.devices:
            self.devices[key] = torch.cat(self.arrays).to(ids.device)
        if ids.max() >= self.size:
            raise ValueError("Map feature IDs do not belong to this policy's feature bank")
        return self.devices[key][ids.clamp_min(0)]


def surface_points(mesh, spacing, tabletop=False):
    """One observed surface candidate per local voxel, with deterministic ordering."""
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
    ordered = [retained[k] for k in sorted(retained)]
    return np.stack([p for p, _ in ordered]), np.stack([n for _, n in ordered])


def sphere_views(count):
    k = np.arange(count)
    z = 1 - 2 * (k + .5) / count
    angle = k * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z * z)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def visual_mesh(body):
    import sapien
    import trimesh
    from mani_skill.utils.geometry.trimesh_utils import (
        get_render_shape_meshes,
        merge_meshes,
    )
    meshes = []
    for shape in body.render_shapes:
        if isinstance(shape, (sapien.render.RenderShapeCylinder, sapien.render.RenderShapeCapsule)):
            # SAPIEN primitives point along X; trimesh primitives point along Z.
            factory = trimesh.creation.cylinder if isinstance(shape, sapien.render.RenderShapeCylinder) else trimesh.creation.capsule
            mesh = factory(height=2 * shape.half_length, radius=shape.radius)
            rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
            mesh.apply_transform(shape.local_pose.to_transformation_matrix() @ rotation)
            meshes.append(mesh)
        else:
            meshes.extend(get_render_shape_meshes(shape))
    return merge_meshes(meshes)


def appearance_key(body, mesh, config, tabletop):
    """Geometry, material, texture contents, extraction settings and DINO provenance."""
    digest = hashlib.sha256()
    digest.update(np.asarray(mesh.vertices, dtype=np.float32).tobytes())
    digest.update(np.asarray(mesh.faces, dtype=np.int32).tobytes())
    settings = {k: config[k] for k in ("voxel_size", "views", "extra_views")}
    settings.update(version=MAP_VERSION, tabletop=tabletop, dino_sha256=config["dino"]["sha256"],
                    source_revision=config["dino"].get("source_revision"), image_size=256)
    digest.update(json.dumps(settings, sort_keys=True).encode())
    for shape in body.render_shapes:
        materials = [part.material for part in shape.parts] if hasattr(shape, "parts") else [shape.material]
        for material in materials:
            for name in ("base_color", "roughness", "metallic", "specular", "emission", "transmission", "ior"):
                value = getattr(material, name, None)
                if value is not None:
                    digest.update(np.asarray(value).tobytes())
            for name in ("base_color_texture", "normal_texture", "roughness_texture", "metallic_texture", "emission_texture"):
                texture = getattr(material, name, None)
                if texture is not None:
                    filename = getattr(texture, "filename", "")
                    if filename and Path(filename).is_file():
                        digest.update(file_hash(filename).encode())
    return digest.hexdigest()


class IsolatedRenderer:
    def __init__(self, body):
        import sapien
        # Render components only: no physics system, collision shapes, or task scene edits.
        self.scene = sapien.Scene([sapien.render.RenderSystem()])
        entity = sapien.Entity()
        clone = body.clone()
        clone.visibility = 1
        entity.add_component(clone)
        self.scene.add_entity(entity)
        self.scene.set_ambient_light([.5, .5, .5])
        self.scene.add_directional_light([0, 0, -1], [.8, .8, .8], shadow=False)
        camera_mount = sapien.Entity()
        shader = str(Path(sapien.__file__).parent / "vulkan_shader/default")
        self.camera = sapien.render.RenderCameraComponent(256, 256, shader)
        self.camera.set_fovy(np.pi / 3, compute_x=True)
        self.camera.near, self.camera.far = .001, 100
        camera_mount.add_component(self.camera)
        self.scene.add_entity(camera_mount)

    def render(self, eye, target):
        from mani_skill.utils.sapien_utils import look_at
        direction = np.asarray(target) - eye
        up = [0, 1, 0] if abs(direction[2]) / np.linalg.norm(direction) > .95 else [0, 0, 1]
        self.camera.entity.set_pose(look_at(eye, target, up).sp)
        self.scene.update_render()
        self.camera.take_picture()
        color = self.camera.get_picture("Color")[..., :3]
        rgb = color if color.dtype == np.uint8 else (color.clip(0, 1) * 255).round().astype(np.uint8)
        depth = -self.camera.get_picture("Position")[..., 2]
        return rgb, depth, self.camera.get_model_matrix(), self.camera.get_intrinsic_matrix()

    def close(self):
        self.scene.clear()


@torch.no_grad()
def observe(renderer, net, xyz, eye, target, device):
    rgb, depth, matrix, intrinsics = renderer.render(eye, target)
    camera = (xyz - matrix[:3, 3]) @ matrix[:3, :3]
    z = -camera[:, 2]
    uv = np.column_stack((camera[:, 0], -camera[:, 1])) / np.maximum(z[:, None], 1e-9)
    uv = uv * np.array([intrinsics[0, 0], intrinsics[1, 1]]) + intrinsics[:2, 2]
    pixels = np.rint(uv).astype(int)
    valid = (z > .001) & (pixels >= 0).all(1) & (pixels < 256).all(1)
    ids = np.flatnonzero(valid)
    ids = ids[np.abs(depth[pixels[ids, 1], pixels[ids, 0]] - z[ids]) < np.maximum(.0015, .002 * z[ids])]
    if not len(ids):
        return ids, np.empty((0, 1024), np.float32)
    tokens = net(torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(device))
    features = tokens.transpose(1, 2).reshape(1, 1024, 16, 16)
    grid = torch.tensor(2 * (uv[ids] + .5) / 256 - 1, device=device, dtype=torch.float32)[None, :, None]
    samples = F.grid_sample(features.float(), grid, align_corners=False, padding_mode="border")
    return ids, samples[0, :, :, 0].T.cpu().numpy()


def build_template(body, mesh, config, destination, net, device, tabletop=False):
    xyz, normals = surface_points(mesh, config["voxel_size"], tabletop)
    sums, counts = np.zeros((len(xyz), 1024), np.float32), np.zeros(len(xyz), np.int32)
    renderer = IsolatedRenderer(body)
    start = time.monotonic()
    used = 0
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


class MapObservationWrapper(gym.Wrapper):
    def __init__(self, env, config, bank=None):
        super().__init__(env)
        self.config = config
        self.bank = bank if bank is not None else FeatureBank()
        self.bindings = []

    def bind(self):
        import sapien
        base = self.unwrapped
        components = [(a, a.name == "table-workspace") for a in base.scene.actors.values()
                      if a.name != "ground" and (a.name != "table-workspace" or self.config["background"] == "table")]
        agents = base.agent.agents if hasattr(base.agent, "agents") else [base.agent]
        for agent in agents:
            names = GRIPPERS.get(agent.uid)
            if self.config["robot"] == "gripper" and names is None:
                raise ValueError(f"No gripper link selection for {agent.uid}")
            components.extend((link, False) for link in agent.robot.links
                              if self.config["robot"] == "full" or link.name in names)
        self.bindings = []
        seen, net = set(), None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            for component, tabletop in components:
                for row, (obj, scene_index) in enumerate(zip(component._objs, component._scene_idxs.tolist())):
                    entity = obj if isinstance(obj, sapien.Entity) else obj.entity
                    if entity in seen:
                        continue
                    seen.add(entity)
                    body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
                    if body is None:
                        continue
                    mesh = visual_mesh(body)
                    if mesh is None:
                        continue
                    key = appearance_key(body, mesh, self.config, tabletop)
                    path = Path(self.config["cache"]) / f"{key}.h5"
                    if not path.is_file():
                        if net is None:
                            # Cache hits/misses must not change the policy/environment RNG stream.
                            with torch.random.fork_rng(devices=[]):
                                net = FrozenDINO(self.config["dino"]).to(device)
                        build_template(body, mesh, self.config, path, net, device, tabletop)
                    xyz, ids = self.bank.add(path)
                    self.bindings.append((component, row, scene_index, xyz.to(base.device), ids.to(base.device)))
        finally:
            del net

    def observation(self):
        base = self.unwrapped
        points, ids = [[] for _ in range(base.num_envs)], [[] for _ in range(base.num_envs)]
        transforms = {}
        for component, row, scene_index, local, features in self.bindings:
            key = id(component)
            if key not in transforms:
                transforms[key] = component.pose.to_transformation_matrix()
            pose = transforms[key][row]
            points[scene_index].append(local @ pose[:3, :3].T + pose[:3, 3])
            ids[scene_index].append(features)
        if any(not p for p in points):
            raise ValueError("Map selection contains no observed components")
        return batch_observations([{"xyz": torch.cat(p)[None], "feature_ids": torch.cat(i)[None]} for p, i in zip(points, ids)])

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        # Reconfiguration replaces actors/links. Bind after every reset; cached DINO is reused.
        self.bind()
        return self.observation(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(), reward, terminated, truncated, info

    def close(self):
        self.bindings.clear()
        self.env.close()
