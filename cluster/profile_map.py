"""Profile the saved map configuration; never update a training checkpoint."""
import argparse
import functools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import jax
import jax.numpy as jp
import numpy as np
import optax
from brax.training import types
from brax.training.agents.ppo import losses
from mujoco_playground._src import wrapper

from benchmark.common.envs import make_env
from benchmark.common.policy import network_factory
from benchmark.common.points import fps, gather, knn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--training-pid', type=int)
    args = parser.parse_args()
    meta = json.loads((Path(args.run) / 'config.json').read_text())
    cfg = meta['ppo']; root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    report = {'ppo': cfg, 'measurements': {}, 'devices': [str(x) for x in jax.devices()],
              'method': 'Synchronized component microbenchmarks; reset observations and fresh parameters. '
                        'Iteration totals are estimates, not an end-to-end training trace.'}

    def save():
        (root / 'timings.json').write_text(json.dumps(report, indent=2))

    def pause():
        if not args.training_pid:
            return None
        # An independent watchdog resumes training even if this profiler crashes.
        code = 'import os,signal,time;time.sleep(20);os.kill(%d,signal.SIGCONT)' % args.training_pid
        watchdog = subprocess.Popen([sys.executable, '-c', code])
        os.kill(args.training_pid, signal.SIGSTOP)
        time.sleep(.3)
        return watchdog

    def resume(watchdog):
        if watchdog:
            os.kill(args.training_pid, signal.SIGCONT)
            watchdog.terminate(); watchdog.wait()

    def measure(name, fn, *inputs, repeat=10):
        t = time.perf_counter()
        compiled = jax.jit(fn).lower(*inputs).compile()
        jax.block_until_ready(compiled(*inputs))
        compile_seconds = time.perf_counter() - t
        watchdog = pause()
        try:
            samples = []
            for _ in range(repeat):
                t = time.perf_counter(); jax.block_until_ready(compiled(*inputs))
                samples.append((time.perf_counter() - t) * 1000)
            stats = {'median_ms': float(np.median(samples)), 'min_ms': min(samples),
                     'compile_and_warmup_seconds': compile_seconds}
            report['measurements'][name] = stats; save(); print(name, stats, flush=True)
        finally:
            resume(watchdog)
        return compiled

    env = make_env(meta['env_config'], cfg['num_envs'])
    wrapped = wrapper.wrap_for_brax_training(env, cfg['episode_length'], cfg['action_repeat'])
    key = jax.random.PRNGKey(900)
    state = jax.jit(wrapped.reset)(jax.random.split(key, cfg['num_envs']))
    jax.block_until_ready(state)
    report.update(points=meta['map_points'], bank_shape=list(env.bank.features.shape),
                  observation_shapes={k:list(v.shape) for k,v in state.obs.items()})
    save(); print('SHAPES', report['observation_shapes'], report['bank_shape'], flush=True)
    action = jp.zeros((cfg['num_envs'], env.action_size))
    measure('physics_and_map_observation', wrapped.step, state, action)
    factory = network_factory('map', cfg['network_factory'], env.bank)
    net = factory(env.observation_size, env.action_size)
    params = losses.PPONetworkParams(net.policy_network.init(key), net.value_network.init(key))
    actor = measure('actor_rollout_batch8', lambda p,o:net.policy_network.apply(None,p,o), params.policy,state.obs)
    xyz = state.obs['xyz'].reshape(cfg['num_envs'],-1,3); valid = state.obs['feature_ids'] >= 0
    measure('fps_256_batch8', lambda x,v:fps(x,v,256),xyz,valid)
    ids,_ = jax.jit(lambda x,v:fps(x,v,256))(xyz,valid); centers=gather(xyz,ids)
    measure('knn_256_batch8',lambda x,c,v:knn(x,c,v,16),xyz,centers,valid)
    measure('bank_projection',lambda b,w:jp.matmul(b,w,precision=jax.lax.Precision.HIGHEST),env.bank.features,jp.ones((env.bank.features.shape[-1],64)))
    obs=jax.tree.map(lambda x:jp.broadcast_to(x[0],(cfg['batch_size'],cfg['unroll_length'],*x.shape[1:])),state.obs)
    shape=(cfg['batch_size'],cfg['unroll_length'])
    data=types.Transition(observation=obs,next_observation=obs,action=jp.zeros((*shape,env.action_size)),
        reward=jp.ones(shape),discount=jp.ones(shape),extras={'state_extras':{'truncation':jp.zeros(shape)},
        'policy_extras':{'raw_action':jp.zeros((*shape,env.action_size)),'log_prob':jp.zeros(shape),
                         'distribution_params':jp.zeros((*shape,net.parametric_action_distribution.param_size))}})
    loss=functools.partial(losses.compute_ppo_loss,ppo_network=net,
        **{k:cfg[k] for k in ('entropy_cost','discounting','reward_scaling','gae_lambda','clipping_epsilon') if k in cfg})
    optim=optax.chain(optax.clip_by_global_norm(cfg['max_grad_norm']),optax.adam(cfg['learning_rate']))
    optstate=optim.init(params)
    def update(p,s,d):
        (value,metrics),grads=jax.value_and_grad(loss,has_aux=True)(p,None,d,key)
        changes,s=optim.update(grads,s,p)
        return optax.apply_updates(p,changes),s,value
    update_compiled=measure('ppo_minibatch_gradient_and_optimizer',update,params,optstate,data)
    timings=report['measurements']; rollout_calls=cfg['batch_size']*cfg['num_minibatches']*cfg['unroll_length']/cfg['num_envs']; updates=cfg['num_minibatches']*cfg['num_updates_per_batch']
    parts={'rollout':rollout_calls*(timings['physics_and_map_observation']['median_ms']+timings['actor_rollout_batch8']['median_ms']),
           'updates':updates*timings['ppo_minibatch_gradient_and_optimizer']['median_ms']}
    report['estimated_iteration_ms']=parts; report['steps_per_iteration']=cfg['batch_size']*cfg['num_minibatches']*cfg['unroll_length'];save()
    print('BREAKDOWN',parts,flush=True)
    watchdog=pause()
    try:
        with jax.profiler.trace(str(root/'trace'),create_perfetto_trace=True):
            for _ in range(3):
                with jax.profiler.TraceAnnotation('actor'):jax.block_until_ready(actor(params.policy,state.obs))
                with jax.profiler.TraceAnnotation('ppo_update'):jax.block_until_ready(update_compiled(params,optstate,data))
    except Exception as error:
        report['trace_error']=str(error);save();print('TRACE',str(error),flush=True)
    finally:
        resume(watchdog)
    print('DONE',flush=True)


if __name__ == '__main__':
    main()
