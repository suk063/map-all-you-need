"""Inspect any Playground manipulation scene with mjviser's browser viewer."""

import argparse
import copy
from pathlib import Path

import benchmark  # noqa: F401  Set runtime defaults before loading JAX/MuJoCo.
import jax
import mujoco
import numpy as np

from benchmark.common.envs import DEFAULT_TASK, TASKS, env_config, make_env
from benchmark.common.mapping import components


def load_scene(env_id, seed=0, map_only=False, background=False, robot="full", *, env=None):
    """Copy the official task reset into a private CPU model for inspection."""
    if env is None:
        env = make_env(env_config(env_id))
    state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
    model = copy.copy(env.mj_model)
    initial = jax.device_get({name: getattr(state.data, name) for name in
                             ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat")})
    if map_only:
        selected = [g for _, geoms, _ in components(model, env_id, {"robot": robot, "background": background}) for g in geoms]
        hidden = np.ones(model.ngeom, dtype=bool)
        hidden[selected] = False
        model.geom_rgba[hidden, 3] = 0
        model.geom_matid[hidden] = -1
        model.site_rgba[:, 3] = 0
        model.tendon_rgba[:, 3] = 0

    def reset(model, data):
        mujoco.mj_resetData(model, data)
        for name, value in initial.items():
            getattr(data, name)[:] = value
        mujoco.mj_forward(model, data)

    data = mujoco.MjData(model)
    reset(model, data)
    return model, data, reset


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=TASKS, default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--map-only", action="store_true", help="Show only map-selected geometry; caches are needed only with --dino-pca")
    parser.add_argument("--map-robot", choices=("full", "gripper"), default="full")
    parser.add_argument("--map-background", choices=("true", "false"), default="false")
    parser.add_argument("--dino-pca", action="store_true", help="Show a DINO PCA-colored map beside the scene")
    parser.add_argument("--map-cache", default=".cache/maps")
    parser.add_argument("--map-views", type=int, default=96)
    parser.add_argument("--map-extra-views", type=int, default=512)
    parser.add_argument("--dino-source", help="Local DINOv3 source for PCA map caches")
    parser.add_argument("--dino-weights", help="Local DINOv3 ViT-L/16 weights")
    return parser


def main():
    cli = parser()
    args = cli.parse_args()
    if args.dino_pca and (args.map_views < 1 or args.map_extra_views < 0):
        cli.error("Map views must be positive and extra views nonnegative")
    from mjviser import Viewer
    import viser

    pca_data = None
    env = None
    if args.dino_pca:
        from benchmark.common.dino import dino_config
        from benchmark.common.mapping import MapObservationWrapper
        from util.dino_pca import PCAView, pca_colors

        config = {"robot": args.map_robot, "background": args.map_background == "true",
                  "voxel_size": .015, "views": args.map_views, "extra_views": args.map_extra_views,
                  "cache": str(Path(args.map_cache).expanduser().resolve()),
                  "dino": dino_config(args.dino_source, args.dino_weights)}
        env = make_env(env_config(args.env_id))
        print("Loading DINO map (missing component caches will be generated)...", flush=True)
        mapped = MapObservationWrapper(env, args.env_id, config)
        features = np.concatenate(mapped.bank.arrays)[np.asarray(mapped.feature_ids)]
        print(f"Fitting scene-wide PCA for {len(features):,} map points...", flush=True)
        pca_data = np.asarray(mapped.local), np.asarray(mapped.body_ids), pca_colors(features)
        del mapped, features

    model, data, reset = load_scene(args.env_id, args.seed, args.map_only, args.map_background == "true", args.map_robot, env=env)
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        render = None
        if pca_data is not None:
            # Rightward in mjviser's default 120-degree camera azimuth. The
            # native model is a private copy; extent affects only initial framing.
            separation = max(model.stat.extent * 1.25, .3)
            offset = np.array([np.sqrt(3) / 2, .5, 0.]) * separation
            pca = PCAView(server, *pca_data, data, offset=offset)
            model.stat.extent *= 1.6
            pca.add_gui(data)

            def render(scene):
                scene.update_from_mjdata(data)
                pca.update(data, scene.fixed_bodies_frame.position)

        viewer = Viewer(model, data, reset_fn=reset, render_fn=render, server=server)
        if args.map_only:
            # mjviser draws fixed planes as grids even when geom alpha is zero.
            from mjviser.conversions import get_body_name
            for geom in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE):
                body = get_body_name(model, int(model.geom_bodyid[geom]))
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom))
                server.scene.remove_by_name(f"/fixed_bodies/{body}/{name}")
        viewer._paused = True  # The pinned Viewer has no public start-paused argument.
        mode = "scene + DINO PCA" if args.dino_pca else "scene"
        print(f"{args.env_id}: http://{args.host}:{args.port} (paused; CPU {mode} inspection)", flush=True)
        viewer.run()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
