"""Export actual map observations for every task as PCA-colored point clouds."""

import argparse
import gc
import json
from pathlib import Path

import benchmark  # noqa: F401  Configure JAX/MuJoCo before imports.
import h5py
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from benchmark.common.dino import dino_config
from benchmark.common.envs import GOAL_BODIES, TASKS, env_config, make_env
from benchmark.common.geometry import scene_bounds
from benchmark.common.mapping import (
    DEFAULT_VOXEL_SIZE, ROBOT_ROOTS, TASK_OBJECTS, MapObservationWrapper, appearance_key,
    components, descendants, visual_mesh,
)
from util.dino_pca import pca_embedding


def collect(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = {"background": args.map_background == "true", "voxel_size": DEFAULT_VOXEL_SIZE,
                "views": args.map_views, "extra_views": args.map_extra_views,
                "cache": str(Path(args.map_cache).resolve()),
                "dino": dino_config(args.dino_source, args.dino_weights)}
    manifest = {"seed": args.seed, "map_settings": settings, "variants": [],
                "coordinate_frame": "normalized_robot",
                "coordinates": "MapObservationWrapper.observation(data, info): robot frame with episode-fixed reset normalization; xyz after one zero-action native step, reset_xyz immediately after native reset",
                "features": "Frozen DINOv3 ViT-L/16, 1024 channels, before trainable projection",
                "pca": "Independent fit per task/mode; PC1/2/3 to RGB with 2nd/98th percentile clipping; no additional downsampling for visualization"}
    modes = ("full", "gripper") if args.map_robot == "both" else (args.map_robot,)
    for task in args.env_id:
        print(f"\n{task}: loading native reset and production map caches", flush=True)
        env = make_env(env_config(task))
        state = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
        stepped = jax.jit(env.step)(state, jnp.zeros(env.action_size))
        model = env.mj_model
        # Compare native reset's cached body transforms with qpos-derived FK.
        # Keep both actual observations; never silently replace the policy input.
        reference = mujoco.MjData(model)
        for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
            getattr(reference, name)[:] = np.asarray(getattr(state.data, name))
        mujoco.mj_forward(model, reference)
        family = next(name for name in ROBOT_ROOTS if task.startswith(name))
        robot_bodies = descendants(model, ROBOT_ROOTS[family])
        object_bodies = descendants(model, TASK_OBJECTS[task])
        for mode in modes:
            config = {**settings, "robot": mode}
            mapped = MapObservationWrapper(env, task, config)
            reset_geometry, _ = mapped.geometry(state.data)
            center, scale = scene_bounds(reset_geometry, mapped.feature_ids >= 0)
            info = {'_map_center': center, '_map_scale': scale}
            obs = jax.device_get(mapped.observation(stepped.data, info))
            reset_obs = jax.device_get(mapped.observation(state.data, info))
            reset_xyz = reset_obs["xyz"].reshape(-1, 3)
            xyz = obs["xyz"].reshape(-1, 3)
            feature_ids = obs["feature_ids"]
            features = np.concatenate(mapped.bank.arrays)[feature_ids]
            print(f"{task}/{mode}: fitting PCA on {len(xyz):,} x {features.shape[1]} input features", flush=True)
            scores, colors, ratios = pca_embedding(features)
            part_ids = np.empty(len(xyz), dtype=np.int32)
            rows, start = [], 0
            for index, ((body, geoms, tabletop), name) in enumerate(zip(
                    components(model, task, config), mapped.component_names, strict=True)):
                key = appearance_key(model, geoms, visual_mesh(model, geoms), config, tabletop)
                cache = Path(config["cache"]) / f"{key}.h5"
                with h5py.File(cache) as f:
                    count = len(f["xyz"])
                    coverage = json.loads(f.attrs["metadata"])["coverage"]
                part_ids[start:start + count] = index
                category = ("robot" if body in robot_bodies else "object" if body in object_bodies else
                            "goal" if model.body(body).name in GOAL_BODIES else "background")
                meshes = [model.mesh(int(model.geom_dataid[g])).name for g in geoms
                          if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
                rows.append({"name": name, "category": category, "points": count, "body_id": body,
                             "geoms": geoms, "meshes": meshes, "cache": str(cache), "coverage": coverage})
                start += count
            assert start == len(xyz) and np.isfinite(xyz).all() and (feature_ids >= 0).all()
            if task.startswith("Aloha"):
                assert all(not {"tabletop", "tablelegs"}.intersection(row["meshes"]) for row in rows)
            filename = f"{task}_{mode}.npz"
            np.savez_compressed(output / filename, xyz=xyz, reset_xyz=reset_xyz, colors=colors, scores=scores.astype(np.float32),
                                feature_ids=feature_ids, body_ids=np.asarray(mapped.body_ids), part_ids=part_ids,
                                normals=obs['normals'].reshape(-1, 3), reset_normals=reset_obs['normals'].reshape(-1, 3),
                                geometry_epsilon=obs['geometry_epsilon'], center=np.asarray(center), scale=np.asarray(scale))
            selected_bodies = np.unique(mapped.body_ids)
            selected_bodies = selected_bodies[model.body_mocapid[selected_bodies] < 0]
            reset_error = np.linalg.norm(np.asarray(state.data.xpos)[selected_bodies] - reference.xpos[selected_bodies], axis=1).max()
            manifest["variants"].append({"task": task, "robot": mode, "points": len(xyz),
                                         "npz": filename, "parts": rows, "variance_ratio": ratios.tolist(),
                                         "step_time_seconds": float(stepped.data.time),
                                         "reset_body_pose_max_error_m": float(reset_error)})
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"SAVED {task}/{mode}: {len(xyz):,} points, {len(rows)} components", flush=True)
            del mapped, features
        del env, state, stepped
        jax.clear_caches()
        gc.collect()
    return output


def render(output):
    from util.map_input_report import render_report

    render_report(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--map-robot", choices=("full", "gripper", "both"), default="both")
    parser.add_argument("--map-background", choices=("true", "false"), default="false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="reports/map-input-pca")
    parser.add_argument("--map-cache", default=".cache/maps")
    parser.add_argument("--map-views", type=int, default=96)
    parser.add_argument("--map-extra-views", type=int, default=512)
    parser.add_argument("--dino-source")
    parser.add_argument("--dino-weights")
    parser.add_argument("--render-only", action="store_true", help="Regenerate HTML/PNG from an existing export")
    args = parser.parse_args()
    if args.map_views < 1 or args.map_extra_views < 0:
        parser.error("Map views must be positive and extra views nonnegative")
    output = Path(args.output).resolve() if args.render_only else collect(args)
    render(output)
    print(f"Report: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
