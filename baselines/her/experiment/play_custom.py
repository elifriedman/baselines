import click
import numpy as np
import pickle, json

from baselines import logger
from baselines.common import set_global_seeds
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=1)
@click.option('--render', type=int, default=1)
@click.option('--steps_past_done', type=int, default=50)
@click.option('--weight_density', type=int, default=0)
@click.option('--param_file', type=str, default="")
@click.option('--start_pos', '-s', multiple=True)
def main(policy_file, seed, n_test_rollouts, render, steps_past_done, weight_density, param_file, start_pos):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    if param_file:
        with open(param_file) as f:
            override_params = json.load(f)
            params.update(override_params)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    env = params['make_env']()
    env.seed(seed)

    data = DataGather(seed, n_test_rollouts, weight_density, start_pos)

    while not data.done():
        obs = env.reset()
        past_done_i = 0

        data.reset(env, obs)
        w = env.env.weights
        while True:
            prev_obs = obs
            o, ag, g = obs['observation'], obs['achieved_goal'], obs['desired_goal']
            try:
                policy_output = policy.get_actions(
                        o, ag, g, w,
                        compute_Q=True,
                        noise_eps=0.,
                        random_eps=0.,
                        use_target_net=params['test_with_polyak'])
            except TypeError:
                policy_output = policy.get_actions(
                        o, ag, g, w,
                        compute_Q=True,
                        noise_eps=0.,
                        random_eps=0.,
                        use_target_net=params['test_with_polyak'])

            u, Q = policy_output
            action = u

            obs, rew, done, info = env.step(u)

            data.increment(prev_obs, action, obs, rew, done, info)
            if render:
                env.render()

            if past_done_i > steps_past_done:
                break

            if done:
                past_done_i += 1

        data.log()

    data.reset(env, obs)
    data.log_all()


import collections

            
class DataGather(object):
    def __init__(self, seed, num_rollouts, weight_density, start_pos):
        self.seed = seed
        self.iteration = 0
        self.data = {}
        self.num_rollouts = num_rollouts
        self.weight_density = weight_density
        self.start_pos = []
        if start_pos:
            for coord in start_pos:
                self.start_pos.append(tuple(float(pos) for pos in coord.split(',')))
        self.weights = None
        self.N = None

    def _get_weights(self, i):
        if self.weights is None or self.weight_density == 0:
            return np.array([1]*self.N + [0]*self.N) 
        else:
            return self.weights[i]

    def done(self):
        return ((self.weight_density == 0 and self.iteration == self.num_rollouts) or 
                (self.weights is not None and self.iteration == len(self.weights)*len(self.start_pos) - 1))

    def reset(self, env, initial_obs):
        self.N = env.env.N
        if self.weights is None:
            weights = np.meshgrid(*[np.linspace(0, 1, self.weight_density)]*(2*self.N - 1))
            weights = np.array([w.reshape(-1) for w in weights]).T
            weights = weights[np.sum(weights, axis=1) <= 1]
            weights = np.concatenate([weights, 1 - weights.sum(axis=1, keepdims=True)], axis=1)
            self.weights = weights
        if not self.start_pos:
            self.start_pos = [(90,)*self.N]

        self.actions = []
        self.observations = []
        self.rewards = []

        # set w
        w_it = self.iteration % len(self.weights)
        s_it = int((self.iteration - w_it) / len(self.weights))
        env.env.weights = self._get_weights(w_it)
        env.env.state = np.array(self.start_pos[s_it] + (0.,)*self.N)
        env.env.SAMPLE_STRATEGY = 'zero'
        initial_obs.update(env.env._make_obs())
        self.iteration += 1

    def increment(self, prev_obs, action, obs, rew, done, info):
        self.actions.append(action)
        self.observations.append(obs['observation'])
        self.rewards.append(rew)

    def log(self):
        it = self.iteration - 1
        w_it = it % len(self.weights)
        s_it = int((it - w_it) / len(self.weights))
        self.data[it] = {
            'w': self._get_weights(w_it),
            'observation': np.array(self.observations),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'start': self.start_pos[s_it]
        }

    def log_all(self):
        import csv, json
        np.save("/usr/local/logs/result_file_{}".format(self.seed), self.data)

if __name__ == '__main__':
    main()
