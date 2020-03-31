import argparse
import os

import gym
import numpy as np
import tensorflow as tf

from dataset import Sal360
from custom_env.envs import CustomEnv
from networks.policy_net import Policy_net
from networks.discriminator import Discriminator
from algo.ppo import PPOTrain
from utils import config


def generator():
    pass


def get_expert_data():
    sal360 = Sal360()
    return sal360.get_expert_trajectory(target_video="10_Cows.mp4")


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e4))
    return parser.parse_args()


def main(args=None):
    env = CustomEnv()

    target_video = "10_Cows.mp4"

    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, 0.95)
    D = Discriminator(env)

    sal360 = Sal360()
    expert_obs, expert_acs, expert_doens = sal360.get_expert_trajectory(target_video=target_video)

    saver = tf.train.Saver()
    logdir = os.path.join(config.log_path, config.GAIL)
    savedir = os.path.join(config.weight_path, config.GAIL)

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        success_num = 0
        expert_obs_iter, expert_acs_iter = iter(expert_obs), iter(expert_acs)
        for iteration in range(10_000):

            obs, expert_acs, target_video = env.reset(trajectory=True, inference=True,
                                                      target_video=target_video, randomness=False)

            observations = []
            actions = []
            try:
                expert_observations, expert_actions = next(expert_obs_iter), next(expert_acs_iter)
            except StopIteration:
                expert_obs_iter, expert_acs_iter = iter(expert_obs), iter(expert_acs)
                expert_observations, expert_actions = next(expert_obs_iter), next(expert_acs_iter)

            # do NOT use rewards to update policy
            rewards = []
            v_preds = []
            run_policy_steps = 0
            while True:
                run_policy_steps += 1

                # (k, 84, 84, 3) --> (1, 8, 84,84,3)  (k : 4~8)
                obs = tf.keras.preprocessing.sequence.pad_sequences([obs], padding='post', value=256, maxlen=8)

                act, v_pred = Policy.act(obs=np.array([obs]), stochastic=True)

                v_pred = v_pred.item()

                next_obs, reward, done, next_ea = env.step(act)

                observations.append(np.array(obs))
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                if done:
                    next_obs = tf.keras.preprocessing.sequence.pad_sequences([next_obs], padding='post', value=256,
                                                                             maxlen=8)
                    _, v_pred = Policy.act(obs=np.array([next_obs]), stochastic=True)
                    v_preds_next = v_preds[1:] + [v_pred.item()]
                    obs, expert_acs, target_video = env.reset(trajectory=True, inference=True,
                                                              target_video=target_video, randomness=False)

                    try:
                        expert_observations, expert_actions = next(expert_obs_iter), next(expert_acs_iter)
                    except StopIteration:
                        expert_obs_iter, expert_acs_iter = iter(expert_obs), iter(expert_acs)
                        expert_observations, expert_actions = next(expert_obs_iter), next(expert_acs_iter)
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            if sum(rewards) >= 195:
                success_num += 1
                if success_num >= 100:
                    saver.save(sess, savedir + '/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + [8] + list(ob_space.shape))
            actions = np.reshape(actions, [-1, 2]).astype(np.float32)

            # train discriminator
            # for i in range(2):
            # for eo, ea, o, a in zip(expert_observations, expert_actions, observations, actions):
            #     D.train(np.array([eo]), np.array([ea]), np.array([o]), np.array([a]))
            D.train(expert_s=expert_observations,
                    expert_a=expert_actions,
                    agent_s=observations,
                    agent_a=actions)
            D.reset_state()
            # output of this discriminator is reward
            d_rewards = D.get_rewards(observations, actions)
            d_rewards = np.reshape(d_rewards, newshape=[-1, 1]).astype(dtype=np.float32)
            gaes = PPO.get_gaes(d_rewards, v_preds, v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # train policy
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            PPO.assign_policy_parameters()
            for ob, ac, gae, dr, vp in zip(observations, actions, gaes, d_rewards, v_preds_next):
                print(np.shape(ob), np.shape(ac), np.shape(gae), np.shape(dr), np.shape(vp))
                PPO.train(np.array([ob]), np.array([ac]), np.array([gae]), np.array([dr]), np.array([vp]))

            PPO.train(obs=observations,
                      actions=actions,
                      gaes=gaes,
                      rewards=d_rewards,
                      v_preds_next=v_preds_next)
            PPO.reset_states()
            for epoch in range(6):
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      gaes=inp[2],
                                      rewards=inp[3],
                                      v_preds_next=inp[4])

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    main()
