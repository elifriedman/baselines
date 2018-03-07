from baselines.ppo1 import mlp_policy, pposgd_simple
from baselines.common import set_global_seeds, tf_util as U, discount
import tensorflow as tf
import gym, roboschool, os, sys
import numpy as np

stochastic=True

env = gym.make("RoboschoolHopper-v1" if len(sys.argv) <= 1 else sys.argv[1])
ob_space = env.observation_space
ac_space = env.action_space

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name,
        ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=2)

def demo_run():
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(0)

    obs = env.reset()

    pi = policy_fn("pi", ob_space, ac_space)

    U.initialize()

    fname = os.path.join("/usr/local/logs", 'final_state')
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), fname)

    np.set_printoptions(suppress=True)
    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()

        obsvs = []
        rewards = []
        vpreds = []
        gvfs = []

        while 1:
            a, vpred, gvf = pi.act(stochastic, obs)

            obs, r, done, _ = env.step(a)

            obsvs.append(obs)
            rewards.append(r)
            vpreds.append(vpred)
            gvfs.append((frame, )+ tuple(g for g in gvf))

            score += r
            frame += 1
            still_open = env.render("human")
            if still_open==False:
                return
            if not done: continue

            discounted = discount(np.array(rewards), 0.99)
            np.savetxt(sys.stdout.buffer, np.array(gvfs))
            np.savetxt(sys.stdout.buffer, np.array(obsvs))

            if restart_delay==0:
                print("score=%0.2f in %i frames" % (score, frame))
                if still_open!=True:      # not True in multiplayer or non-Roboschool environment
                    break
                restart_delay = 60*2  # 2 sec at 60 fps
            restart_delay -= 1
            if restart_delay==0: break

if __name__=="__main__":
    demo_run()
