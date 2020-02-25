import argparse
import gym
import numpy as np
import tensorflow as tf
from networks.policy_net import Policy_net
from algo.ppo import PPOTrain
from custom_env.envs import CustomEnv
from utils.config import log_path, weight_path


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo')
    parser.add_argument('--savedir', help='save directory', default='trained_models/ppo')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1e4), type=int)
    return parser.parse_args()


gamma = 0.95
iterations = 1000


def main(args):
    # env = gym.make('CartPole-v0')
    # env.seed(0)
    env = CustomEnv()
    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=gamma)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(log_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        obs, acs, target_video = env.reset()
        success_num = 0

        for iteration in range(iterations):
            observations = []
            actions = []
            pred_actions = []
            rewards = []
            v_preds = []
            episode_length = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                episode_length += 1
                obs = np.array([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                acs = np.array([acs]).astype(dtype=np.float32)
                pred_act, v_pred = Policy.act(obs=obs, acs=acs, stochastic=True)

                # act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                next_obs, reward, done, info = env.step(acs)

                observations.append(obs)
                actions.append(acs)
                pred_actions.append(pred_act)
                rewards.append(reward)
                v_preds.append(v_pred)

                if done:
                    next_obs = np.stack([next_obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                    _, v_pred = Policy.act(obs=next_obs, stochastic=True)
                    v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                    obs, acs, target_video = env.reset()
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            if sum(rewards) >= 195:
                success_num += 1
                if success_num >= 100:
                    saver.save(sess, weight_path + '/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # convert list to numpy array for feeding tf.placeholder
            # observations = np.reshape(observations, newshape=[-1,] + list(ob_space.shape))
            observations = np.array(observations).astype(dtype=np.float32)

            actions = np.array(actions).astype(dtype=np.float32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            PPO.assign_policy_parameters()

            inp = [observations, actions, gaes, rewards, v_preds_next]

            # train
            for epoch in range(6):
                # sample indices from [low, high)
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
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
    args = argparser()
    main(args)
