import argparse
import os

import gym
import numpy as np
import tensorflow as tf

from custom_env.envs import CustomEnv
from networks.policy_net import Policy_net
from networks.discriminator import Discriminator
from algo.ppo import PPOTrain
from utils import config



def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e4))
    return parser.parse_args()


def main(args=None):
    env = CustomEnv()

    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, 0.95)
    D = Discriminator(env)

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

        for iteration in range(10_000):

            obs, expert_acs, target_video = env.reset(trajectory=True)

            observations = []
            actions = []
            expert_observations, expert_actions = [], []

            # do NOT use rewards to update policy
            rewards = []
            v_preds = []
            run_policy_steps = 0
            while True:
                run_policy_steps += 1
                act, v_pred = Policy.act(obs=np.array([obs]), stochastic=True)

                v_pred = v_pred.item()
                next_obs, reward, done, next_ea = env.step(expert_acs)

                expert_observations.append(np.array(obs))
                expert_actions.append(expert_acs)
                observations.append(np.array(obs))
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                if done:
                    _, v_pred = Policy.act(obs=np.array([next_obs]), stochastic=True)
                    v_preds_next = v_preds[1:] + [v_pred.item()]
                    break
                else:
                    obs = next_obs
                    expert_acs = next_ea
            print(np.shape(observations), np.shape(actions), np.shape(expert_observations), np.shape(expert_actions))
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
            # observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            # actions = np.array(actions).astype(dtype=np.int32)

            # train discriminator
            for i in range(2):
                D.train(expert_s=expert_observations,
                        expert_a=expert_actions,
                        agent_s=observations,
                        agent_a=actions)

            # output of this discriminator is reward
            d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # train policy
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            PPO.assign_policy_parameters()
            PPO.train(obs=observations,
                      actions=actions,
                      gaes=gaes,
                      rewards=d_rewards,
                      v_preds_next=v_preds_next)
            # for epoch in range(6):
            #     sample_indices = np.random.randint(low=0, high=observations.shape[0],
            #                                        size=32)  # indices are in [low, high)
            #     sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
            #     PPO.train(obs=sampled_inp[0],
            #               actions=sampled_inp[1],
            #               gaes=sampled_inp[2],
            #               rewards=sampled_inp[3],
            #               v_preds_next=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      gaes=inp[2],
                                      rewards=inp[3],
                                      v_preds_next=inp[4])

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    main()