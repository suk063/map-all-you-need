"""Inspect any Playground manipulation scene with mjviser's browser viewer."""

import argparse
import copy

import benchmark  # noqa: F401  Set runtime defaults before loading JAX/MuJoCo.
import jax
import mujoco
import numpy as np

from benchmark.common.envs import DEFAULT_TASK, TASKS, env_config, make_env
from benchmark.common.mapping import components


def load_scene(env_id, seed=0, map_only=False, background=False, robot="full"):
    """Copy the official task reset into a private CPU model for inspection."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=TASKS, default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--map-only", action="store_true", help="Show map-selected geometry, without building DINO caches")
    parser.add_argument("--map-robot", choices=("full", "gripper"), default="full")
    parser.add_argument("--map-background", choices=("true", "false"), default="false")
    args = parser.parse_args()
    from mjviser import Viewer
    import viser

    model, data, reset = load_scene(args.env_id, args.seed, args.map_only, args.map_background == "true", args.map_robot)
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        viewer = Viewer(model, data, reset_fn=reset, server=server)
        if args.map_only:
            # mjviser draws fixed planes as grids even when geom alpha is zero.
            from mjviser.conversions import get_body_name
            for geom in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE):
                body = get_body_name(model, int(model.geom_bodyid[geom]))
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom))
                server.scene.remove_by_name(f"/fixed_bodies/{body}/{name}")
        viewer._paused = True  # The pinned Viewer has no public start-paused argument.
        print(f"{args.env_id}: http://{args.host}:{args.port} (paused; CPU scene inspection)", flush=True)
        viewer.run()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
