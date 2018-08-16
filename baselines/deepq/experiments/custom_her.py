import os
import time
import argparse
import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import BatchInput, load_state, save_state
from baselines.deepq.simple import ActWrapper
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    n_hid = 256
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=n_hid, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=n_hid, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=n_hid, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def build_input_maker(env):
    obs = env.reset()
    shape = obs['achieved_goal'].shape
    obslen = shape[0] // 2
    def input_maker(obs, weight=None):
        obs_actual = obs['observation']
        weight = obs['achieved_goal'][obslen:] if weight is None else weight
        goal = obs['desired_goal'][:obslen]
        return np.concatenate([obs_actual, weight, goal])
    test = input_maker(obs)
    return input_maker, test.shape


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', help='environment ID', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--discretization', help='# actions', type=int, default=3)
    parser.add_argument('--num_cpu', help='# cpus', type=int, default=1)
    parser.add_argument('--n_timesteps', help='# timesteps', type=int, default=1)
    parser.add_argument('--logdir', help='log directory', type=str, default="logs/")
    parser.add_argument('--save_freq', help='after how many episodes to save', type=int, default=150)
    parser.add_argument('--replay_k', help='how much augmentation', type=int, default=0)
    args = parser.parse_args()
    set_global_seeds(args.seed)
    logger.configure(dir=args.logdir)


    action_choices = np.linspace(-1, 1, args.discretization)
    with U.make_session(args.num_cpu):
        # Create the environment
        env = gym.make(args.env_name)
        input_maker, input_shape = build_input_maker(env)

        # Create all the functions necessary to train the model
        obs_ph = lambda name: BatchInput(input_shape, name=name)
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=obs_ph,
            q_func=model,
            num_actions=args.discretization,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )

        act_params = {
            "make_obs_ph": obs_ph,
            "q_func": model,
            "num_actions": args.discretization,
            "env": args.env_name,
        }
        act = ActWrapper(act, act_params)

        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        best_reward = -np.inf
        obs = env.reset()

        start_time = time.time()
        epoch_time = time.time()
        for t in itertools.count():
            if t >= args.n_timesteps:
                act.save(os.path.join(args.logdir, "policy_latest.pkl"))
                logger.record_tabular("duration_total", time.time() - start_time)
                logger.record_tabular("duration_epoch", time.time() - epoch_time)
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
                break

            # Take action and update exploration to the newest value
            input_obs = input_maker(obs)
            action = act(input_obs[None], update_eps=exploration.value(t))[0]
            action = action_choices[action]
            new_obs, rew, done, _ = env.step(action)
            new_input_obs = input_maker(new_obs)
            # Store transition in the replay buffer.
            replay_buffer.add((input_obs, action, rew, new_input_obs, float(done)))

            for k in range(args.replay_k):
                w = env.env.sample_weights()
                info = {
                    "weights": w,
                    "action": action,
                }

                rew = env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
                input_obs = input_maker(obs, w)
                new_input_obs = input_maker(new_obs, w)
                replay_buffer.add((input_obs, action, rew, new_input_obs, float(done)))

            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                if len(episode_rewards) >= args.save_freq:
                    mean_reward = np.mean(episode_rewards[-args.save_freq:])
                    if mean_reward > best_reward + 3:
                        act.save(os.path.join(args.logdir, "policy_best.pkl"))
                        logger.log("Saving new best policy because reward is {}".format(mean_reward))
                        best_reward = mean_reward
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % args.save_freq == 0:
                new_epoch_time = time.time()
                act.save(os.path.join(args.logdir, "policy_{}.pkl".format(len(episode_rewards))))
                logger.record_tabular("duration_total", new_epoch_time - start_time)
                logger.record_tabular("duration_epoch", new_epoch_time - epoch_time)
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
                epoch_time = new_epoch_time


if __name__ == '__main__':
    main()