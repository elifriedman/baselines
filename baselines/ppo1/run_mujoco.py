#!/usr/bin/env python3

from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import tensorflow as tf

def train(env_id, num_timesteps, seed, logdir):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)
    import datetime
    start_time = datetime.datetime.now()
    try:
        pposgd_simple.learn(env, policy_fn,
                max_timesteps=num_timesteps,
                timesteps_per_actorbatch=2048,
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                gamma=0.99, lam=0.95, schedule='linear',
            )
    except:
        pass
    import os
    fname = os.path.join(logdir, 'final_state')
    os.path.join(logdir, 'final_state')
    saver = tf.train.Saver()
    saver.save(tf.get_default_session(), fname)

    env.close()
    print()
    print("Started:", start_time)
    print("Finished:", datetime.datetime.now())

def main():
    parser = mujoco_arg_parser()
    parser.add_argument('--logdir', help='log directory', type=str, default=None)
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, logdir=args.logdir)

if __name__ == '__main__':
    main()
