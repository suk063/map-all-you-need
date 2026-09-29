"""Evaluation-only success definitions and native-step episode statistics."""

import jax.numpy as jp
from mujoco_playground._src.wrapper import Wrapper

VERSION = 'task_success_v1'
DEFINITIONS = {
    'PandaPickCube': 'Benchmark: object-target distance < 0.05 m.',
    'PandaPickCubeOrientation': 'Benchmark: object-target distance < 0.05 m and rotation error < 15 degrees.',
    'PandaOpenCabinet': 'Benchmark: handle-target distance < 0.05 m (target sampled by native task).',
    'AlohaHandOver': 'Benchmark: box-target < 0.05 m, right gripper-box bottom < 0.03 m, left gripper-box top > 0.05 m, box height > 0.05 m.',
    'AlohaSinglePegInsertion': 'Benchmark: peg tip-socket rear < 0.005 m and tip distance to socket axis < 0.005 m.',
    'PandaPickCubeCartesian': 'Native reward/success; native success_threshold and sample mode.',
    'PandaRobotiqPushCube': 'Native success; 0.03 m position, 10 degrees orientation and native hold-duration requirement.',
    'LeapCubeReorient': 'Native reward/success before goal resampling; native success_threshold.',
    'LeapCubeRotateZAxis': 'Benchmark: net integrated positive-z angular displacement >= 2*pi rad before dropping.',
    'AeroCubeRotateZAxis': 'Benchmark: net integrated positive-z angular displacement >= 2*pi rad before dropping.',
}


def rotation_error(a, b):
    return jp.arccos(jp.clip((jp.trace(a.T @ b) - 1) / 2, -1, 1))


class EvaluationWrapper(Wrapper):
    """Accumulate before repetition/autoreset, freezing at the first native done."""

    def __init__(self, env, task):
        super().__init__(env)
        self.task = task
        if task not in DEFINITIONS:
            raise ValueError('Missing success definition: ' + task)
        native = env
        while isinstance(native, Wrapper):
            native = native.env
        self.native = native

    def measurements(self, state, previous):
        env, task, data = self.native, self.task, state.data
        target_data = previous.data
        result = {}
        if task.endswith('RotateZAxis'):
            speed = env.get_cube_angvel(data)[2]
            result['angular_velocity_z_rad_s'] = speed
            result['cube_height_m'] = env.get_cube_position(data)[2]
            success = jp.array(False)  # Net displacement is accumulated below.
        elif task == 'LeapCubeReorient':
            result['orientation_error_rad'] = env._cube_orientation_error(
                data.replace(mocap_quat=target_data.mocap_quat))
            success = state.metrics['reward/success'] > 0
        elif task == 'AlohaSinglePegInsertion':
            tip = data.site_xpos[env._peg_end2_site]
            rear = data.site_xpos[env._socket_rear_site]
            entrance = data.site_xpos[env._socket_entrance_site]
            axis = rear - entrance
            direction = axis / jp.maximum(jp.linalg.norm(axis), 1e-8)
            offset = tip - entrance
            lateral = jp.linalg.norm(offset - jp.dot(offset, direction) * direction)
            error = jp.linalg.norm(tip - rear)
            result.update(position_error_m=error, insertion_lateral_error_m=lateral)
            success = (error < 0.005) & (lateral < 0.005)
        else:
            body = env._box_body if task == 'AlohaHandOver' else env._obj_body
            position = data.xpos[body]
            target = (target_data.mocap_pos[env._mocap_target] if task == 'PandaRobotiqPushCube'
                      else previous.info['target_pos'])
            error = jp.linalg.norm(position - target)
            result['position_error_m'] = error
            success = error < 0.05
            if task in ('PandaPickCubeOrientation', 'PandaRobotiqPushCube'):
                # Mocap xmat can lag changes to the target; use its quaternion directly.
                from mujoco.mjx._src import math
                angle = rotation_error(data.xmat[body], math.quat_to_mat(target_data.mocap_quat[env._mocap_target]))
                result['orientation_error_rad'] = angle
                success &= angle < jp.pi / 12
            if task == 'AlohaHandOver':
                right = jp.linalg.norm(data.site_xpos[env._right_gripper_site] - data.site_xpos[env._box_bottom_site])
                left = jp.linalg.norm(data.site_xpos[env._left_gripper_site] - data.site_xpos[env._box_top_site])
                result.update(right_gripper_error_m=right, left_gripper_error_m=left)
                success &= (right < 0.03) & (left > 0.05) & (position[2] > 0.05)
            elif task == 'PandaPickCubeCartesian':
                success = state.metrics['reward/success'] > 0
            elif task == 'PandaRobotiqPushCube':
                success = state.metrics['success'] > 0
        return success, result

    def reset(self, rng):
        state = self.env.reset(rng)
        _, measures = self.measurements(state, state)
        stats = {k: jp.array(0., dtype=state.reward.dtype) for k in (
            'native_steps', 'return', 'success_once', 'success_final', 'success_count',
            'success_steps', 'terminated', 'nonfinite', 'out_of_bounds_once',
            'action_l2_sum', 'action_delta_l2_sum', 'action_saturation_sum', 'rotation_z_rad')}
        stats['first_success_seconds'] = jp.array(-1., dtype=state.reward.dtype)
        for key in measures:
            for prefix in ('sum/', 'final/'):
                stats[prefix + key] = jp.zeros_like(state.reward)
            stats['min/' + key] = jp.array(jp.inf, dtype=state.reward.dtype)
            stats['max/' + key] = jp.array(-jp.inf, dtype=state.reward.dtype)
        for key in state.metrics:
            stats['native_sum/' + key] = jp.zeros_like(state.reward)
            stats['native_final/' + key] = jp.zeros_like(state.reward)
            stats['native_max/' + key] = jp.array(-jp.inf, dtype=state.reward.dtype)
        info = {**state.info, '_eval_stats': stats, '_eval_action': jp.zeros(self.action_size)}
        return state.replace(info=info)

    def step(self, state, action):
        old = state.info['_eval_stats']
        previous_action = state.info['_eval_action']
        # Native tasks mutate info/metrics; retain the old goal and accumulator.
        previous = state.replace(info=dict(state.info), metrics=dict(state.metrics))
        next_state = self.env.step(state, action)
        success, measures = self.measurements(next_state, previous)
        finite = jp.all(jp.isfinite(next_state.data.qpos)) & jp.all(jp.isfinite(next_state.data.qvel)) & jp.isfinite(next_state.reward)
        stats = dict(old)
        stats['native_steps'] += 1
        stats['rotation_z_rad'] += measures.get('angular_velocity_z_rad_s', 0.) * self.native.dt
        if self.task.endswith('RotateZAxis'):
            success = (stats['rotation_z_rad'] >= 2 * jp.pi) & ~next_state.done.astype(bool)
        success &= finite
        stats['return'] += next_state.reward
        stats['success_once'] = jp.maximum(old['success_once'], success)
        stats['success_final'] = success.astype(next_state.reward.dtype)
        stats['success_steps'] += success
        stats['success_count'] += success & (old['success_final'] == 0)
        stats['first_success_seconds'] = jp.where(success & (old['success_once'] == 0), stats['native_steps'] * self.native.dt, old['first_success_seconds'])
        stats['terminated'] = jp.maximum(old['terminated'], next_state.done)
        stats['nonfinite'] = jp.maximum(old['nonfinite'], ~finite)
        stats['out_of_bounds_once'] = jp.maximum(old['out_of_bounds_once'], next_state.metrics.get('out_of_bounds', 0.))
        stats['action_l2_sum'] += jp.linalg.norm(action)
        stats['action_delta_l2_sum'] += jp.linalg.norm(action - previous_action)
        stats['action_saturation_sum'] += jp.mean(jp.abs(action) >= 0.99)
        for key, value in measures.items():
            stats['sum/' + key] += value
            stats['final/' + key] = value
            stats['min/' + key] = jp.minimum(old['min/' + key], value)
            stats['max/' + key] = jp.maximum(old['max/' + key], value)
        for key, value in next_state.metrics.items():
            stats['native_sum/' + key] += value
            stats['native_final/' + key] = value
            stats['native_max/' + key] = jp.maximum(old['native_max/' + key], value)
        stats = {k: jp.where(old['terminated'] > 0, old[k], value) for k, value in stats.items()}
        info = {**next_state.info, '_eval_stats': stats, '_eval_action': action}
        return next_state.replace(info=info, done=jp.maximum(next_state.done, old['terminated']))
