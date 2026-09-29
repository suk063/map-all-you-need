"""One deferred RGB observation path over unmodified native task dynamics."""

import copy
from pathlib import Path

import jax
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper


def rgb_config(task):
    if task.startswith('Aloha'):
        camera = 'overhead_cam'
    elif task == 'PandaPickCubeCartesian':
        camera = 'front'
    elif task == 'PandaOpenCabinet' or task.startswith(('Leap', 'Aero')):
        camera = 'side'
    else:
        camera = 'benchmark_default'
    return {'backend': 'mjx_warp', 'camera': camera, 'resolution': [64, 64],
            'camera_source': 'scene_default' if camera == 'benchmark_default' else 'native',
            'use_textures': True, 'use_shadows': False, 'enabled_geom_groups': [0, 1, 2]}


def render_model(env, settings):
    """Build a private camera model; native physics and model IDs stay untouched."""
    original = env.mj_model
    if settings['camera_source'] == 'native':
        model = copy.copy(original)
        model.camera(settings['camera'])  # Fail clearly if a pinned asset changes.
        return model

    # Obtain the scene's actual default free view with MuJoCo's own projection
    # rules, including XML statistic center/extent and visual azimuth/elevation.
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(original, camera)
    data = mujoco.MjData(original)
    mujoco.mj_forward(original, data)
    scene = mujoco.MjvScene(original, maxgeom=max(1000, original.ngeom * 2))
    mujoco.mjv_updateScene(original, data, mujoco.MjvOption(), None, camera,
                          mujoco.mjtCatBit.mjCAT_ALL, scene)
    view = mujoco.mjv_averageCamera(*scene.camera)
    forward, up = np.asarray(view.forward), np.asarray(view.up)
    rotation = np.column_stack((np.cross(forward, up), up, -forward))
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    # Native tasks load an XML string with includes in their virtual asset map.
    # MjSpec separates XML includes from binary assets, unlike MjModel's loader.
    includes = {name: content for name, content in env._model_assets.items() if name.endswith('.xml')}
    spec = mujoco.MjSpec.from_string(Path(env.xml_path).read_text(), include=includes, assets=env._model_assets)
    spec.worldbody.add_camera(name=settings['camera'], pos=view.pos, quat=quat,
                              fovy=original.vis.global_.fovy)
    model = spec.compile()
    # A second compile may relocate name-buffer offsets after adding a camera.
    # Compare actual indexed identities, never raw name offsets or whole models.
    for kind, count in (('body', original.nbody), ('geom', original.ngeom), ('joint', original.njnt),
                        ('site', original.nsite), ('mesh', original.nmesh), ('material', original.nmat)):
        accessor = getattr(original, kind)
        other = getattr(model, kind)
        if any(accessor(i).name != other(i).name for i in range(count)):
            raise ValueError(f'Render model {kind} identity differs from native model')
    for field in ('nq', 'nv', 'nbody', 'ngeom', 'njnt', 'nmocap', 'nmesh', 'nmat', 'nlight'):
        if getattr(model, field) != getattr(original, field):
            raise ValueError(f'Render model {field} differs from native model')
    for field in ('body_parentid', 'body_jntadr', 'body_jntnum', 'body_mocapid',
                  'jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'geom_bodyid', 'geom_dataid'):
        if not np.array_equal(getattr(model, field), getattr(original, field)):
            raise ValueError(f'Render model kinematic layout differs: {field}')
    # Preserve constructor changes (including goal hiding and robot materials)
    # on the private model. Name buffers are deliberately excluded.
    for field in dir(original):
        if field.startswith(('body_', 'jnt_', 'geom_', 'site_', 'mesh_', 'mat_', 'tex_', 'light_')) or field == 'qpos0':
            source, destination = getattr(original, field), getattr(model, field)
            if isinstance(source, np.ndarray) and isinstance(destination, np.ndarray):
                if source.shape != destination.shape:
                    raise ValueError(f'Render model array shape differs: {field}')
                destination[:] = source
    return model


class RGBObservationWrapper(Wrapper):
    """Render once after official repetition/autoreset; native obs feed history."""

    def __init__(self, env, settings, num_envs):
        from benchmark.common.envs import GOAL_BODIES

        super().__init__(env)
        self.render_metadata = copy.deepcopy(settings)
        self.render_metadata.update(nworld=num_envs, deferred=True, native_vision=False)
        self._render_model = render_model(env, settings)
        self._render_mjx_model = mjx_env.put_model(self._render_model, impl='warp')
        sizes = {key: env._config[key] for key in ('naconmax', 'naccdmax', 'njmax') if key in env._config}
        self._render_initial = mjx.make_data(self._render_model, impl='warp', **sizes)
        self._hidden_mocap_ids = np.array([
            self._render_model.body_mocapid[i] for i in range(self._render_model.nbody)
            if self._render_model.body(i).name in GOAL_BODIES and self._render_model.body_mocapid[i] >= 0
            and np.any(self._render_model.geom_bodyid == i)
            and np.all(self._render_model.geom_group[self._render_model.geom_bodyid == i] == 5)
        ], dtype=np.int32)
        self._camera_id = self._render_model.camera(settings['camera']).id
        self.render_metadata.update(camera_id=self._camera_id,
                                    camera_position=self._render_model.cam_pos[self._camera_id].tolist(),
                                    camera_quaternion=self._render_model.cam_quat[self._camera_id].tolist(),
                                    camera_fovy=float(self._render_model.cam_fovy[self._camera_id]))
        active = [i == self._camera_id for i in range(self._render_model.ncam)]
        self._rc = mjx.create_render_context(
            mjm=self._render_model, nworld=num_envs, cam_res=tuple(settings['resolution']),
            cam_active=active, render_rgb=active, render_depth=[False] * len(active),
            use_textures=settings['use_textures'], use_shadows=settings['use_shadows'],
            enabled_geom_groups=settings['enabled_geom_groups'])
        self._rc_pytree = self._rc.pytree()

    @property
    def unwrapped(self):
        # Upstream DeferredVisionWrapper calls self.unwrapped.render_state.
        # Delegating would expose Cartesian's *native* vision=False render hook.
        return self

    @property
    def observation_size(self):
        return {'pixels/view_0': (*self.render_metadata['resolution'], 3)}

    def defer_rendering(self):
        pass  # reset/step always preserve native observations until outer render.

    def reset(self, rng):
        state = self.env.reset(rng)
        return state.replace(info={**state.info, '_rgb_render_token': self._render_initial._impl._jax_token})

    def render_state(self, state):
        def prepare(native, token):
            # Some native JAX tasks reset mocap arrays from integer literals;
            # Warp's FFI requires the rendering model's floating-point dtype.
            mocap_pos = native.mocap_pos.astype(self._render_initial.mocap_pos.dtype)
            mocap_quat = native.mocap_quat.astype(self._render_initial.mocap_quat.dtype)
            if len(self._hidden_mocap_ids):
                # Hidden mocap markers must not leak the goal through a light's
                # TRACKCOM motion. Freeze only their private render poses.
                ids = self._hidden_mocap_ids
                mocap_pos = mocap_pos.at[ids].set(self._render_initial.mocap_pos[ids])
                mocap_quat = mocap_quat.at[ids].set(self._render_initial.mocap_quat[ids])
            data = self._render_initial.replace(qpos=native.qpos, qvel=native.qvel, ctrl=native.ctrl,
                                                 mocap_pos=mocap_pos, mocap_quat=mocap_quat)
            data = data.tree_replace({'_impl._jax_token': token})
            # Private forward computes current camera/light/geometry poses for
            # either native backend. It never advances or replaces native data.
            return mjx.forward(self._render_mjx_model, data)

        data = jax.vmap(prepare)(state.data, state.info['_rgb_render_token'])
        data = mjx.refit_bvh(self._render_mjx_model, data, self._rc_pytree)
        packed, _, rendered = mjx.render(self._render_mjx_model, data, self._rc_pytree)
        # RenderContext compacts enabled cameras; our sole view has packed ID 0
        # even when its native model ID is nonzero (Aloha overhead_cam is 2).
        pixels = mjx.get_rgb(self._rc_pytree, 0, packed)
        info = {**state.info, '_rgb_render_token': rendered._impl._jax_token}
        return state.replace(obs={'pixels/view_0': pixels}, info=info)
