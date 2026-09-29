"""Shared map geometry contract and episode-local, isotropic bounds."""

EPSILON_M = 1e-6
ROBOT_FRAMES = {
    'Panda': ('link0',),
    'Aloha': ('left/base_link', 'right/base_link'),
    'Leap': ('leap_mount',),
    'Aero': ('tetheria_mount',),
}


def geometry_config(env_id):
    roots = next((roots for family, roots in ROBOT_FRAMES.items() if env_id.startswith(family)), None)
    if roots is None:
        raise ValueError(f'Unknown robot frame: {env_id}')
    return {'encoding': 'contact_robot_v1',
            'frame': {'origin_bodies': list(roots), 'axes_body': roots[0]},
            'normalization': {'range': [-1, 1], 'isotropic': True, 'update': 'reset',
                              'clip': False, 'scale_floor_m': EPSILON_M},
            'epsilon_m': EPSILON_M, 'epsilon_input': 'geometry_epsilon'}


def validate_geometry(config):
    if config['obs_mode'] == 'map' and (not config.get('map_geometry') or
                                      config['map_geometry'] != geometry_config(config['env_id'])):
        raise ValueError('Incompatible map geometry; retrain with contact_robot_v1 (old map checkpoints cannot resume/evaluate)')


def scene_bounds(xyz, valid):
    """Return one center/scale, ignoring padded points; never stretch axes."""
    import jax.numpy as jnp

    low = jnp.where(valid[..., None], xyz, jnp.inf).min(axis=-2)
    high = jnp.where(valid[..., None], xyz, -jnp.inf).max(axis=-2)
    present = valid.any(axis=-1, keepdims=True)
    low, high = jnp.where(present, low, 0), jnp.where(present, high, 0)
    return (low + high) / 2, jnp.maximum((high - low).max(axis=-1, keepdims=True) / 2, EPSILON_M)
