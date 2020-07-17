import os
import datetime

import argparse
import random

import gym
import numpy as np
import tensorflow as tf

from custom_env.envs import CustomEnv
from networks.discriminator import Discriminator
from networks.policy_net import Policy_net
from algo.ppo import PPOTrain


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo')
    parser.add_argument('--savedir', help='save directory', default='trained_models/ppo')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1e3), type=int)
    return parser.parse_args()

def get_expert_trajectory(target_video, index, env):
    expert_ob, expert_ac = [], []
    for i in range(index):
        idx = random.randint(0, 44)
        x, y = env.dataset.get_expert_train(target_video, idx)
        expert_ob.append(x)
        expert_ac.append(y)
    return expert_ob, expert_ac

def main(args):
    env = CustomEnv()
    writer_path = "log/gail"
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)
    writer = tf.summary.create_file_writer(writer_path)
    writer.set_as_default()

    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy)
    D = Discriminator(env)
    target_video = "11_Abbottsford.mp4"

    # expert_ob, expert_ac = get_expert_trajectory(target_video, 4, env)

    obs, ac, target_video = env.reset(trajectory=False, fx=1, fy=1, saliency=False, inference=False,
                                      target_video=target_video)

    # reward = 0
    success_num = 0
    render = False
    agent_ob = []
    agent_ac = []
    global_step = 0
    for iteration in range(10000):
        observations = []
        actions = []
        v_preds = []
        # rewards = []
        episode_length = 0
        while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
            episode_length += 1
            # obs = np.array([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
            act, v_pred = Policy.act(obs=np.array([obs]), training=False)

            observations.append(obs)
            actions.append(act)
            v_preds.append(v_pred)
            # rewards.append(reward)

            next_obs, reward, done, info = env.step(act)
            if render:
                env.render()
            if done:
                next_obs = np.array([next_obs]).astype(dtype=np.float32)
                _, v_pred = Policy.act(obs=next_obs, training=False)
                v_preds_next = v_preds[1:] + [float(v_pred)]  # next state of terminate state has 0 state value
                obs, ac, target_video = env.reset(trajectory=False, fx=1, fy=1, saliency=False, inference=False,
                                                  target_video=target_video)
                reward = 0
                agent_ob.append(observations)
                agent_ac.append(np.reshape(actions, [-1, 2]))
                global_step += 1
                break
            obs = next_obs
        # tf.summary.scalar("episode_reward", sum(rewards), iteration)

        # if sum(rewards) >= 195:
        #     success_num += 1
            # render = True
        if iteration % 25 == 0:
            D.save()
            PPO.save()
        if iteration != 0 and iteration % 4 == 0:
            expert_ob, expert_ac = get_expert_trajectory(target_video, 4, env)
            print(np.shape(expert_ob), np.shape(expert_ac))
            print(np.shape(agent_ob), np.shape(agent_ac))
            D.train(expert_s=expert_ob,
                    expert_a=expert_ac,
                    agent_s=agent_ob,
                    agent_a=agent_ac)
            agent_ob = []
            agent_ac = []
            d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            observations = tf.keras.applications.mobilenet_v2.preprocess_input(observations, data_format=tf.float32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            PPO.assign_policy_parameters()
            # train
            for epoch in range(6):
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])


if __name__ == '__main__':
    args = argparser()
    main(args)
