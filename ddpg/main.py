#!/usr/bin/env python3
import argparse
import random
from collections import deque
import numpy as np
import torch
import gym
from memory import Memory
from noise import *
from ddpg import DDPG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(env, nb_epoch, nb_epoch_cycle, normalize_observations, 
          actor_lr, critic_lr, action_noise, gamma, nb_train_step, nb_rollout_step, batch_size, memory, tau=0.01):
    max_action = env.action_space.high
    agent = DDPG(memory, env.observation_space.shape[0], env.action_space.shape[0],
                 gamma=gamma, tau=tau, normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise,
                 actor_lr=actor_lr, critic_lr=critic_lr)
    agent.cuda(device)

    step = 0
    episode = 0
    episode_rewards_history = deque(maxlen=100)

    agent.reset()
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_step = 0
    episodes = 0
    t = 0

    epoch = 0
    
    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0

    for epoch in range(nb_epoch):
        for cycle in range(nb_epoch_cycle):
            for t_rollout in range(nb_rollout_step):
                action, q = agent.policy(obs, apply_noise=True, compute_Q=True)
                new_obs, reward, done, info = env.step(max_action * action)
                env.render()
                t += 1
                episode_reward += reward
                episode_step += 1

                epoch_actions.append(action)
                epoch_qs.append(q)
                agent.store_transition(obs, action, reward, new_obs, done)
                obs = new_obs

                if done:
                    epoch_episode_rewards.append(episode_reward)
                    episode_rewards_history.append(episode_reward)
                    epoch_episode_steps.append(episode_step)
                    episode_reward = 0
                    episode_step = 0
                    epoch_episodes += 1
                    episodes += 1

                    agent.reset()
                    obs = env.reset()
            
            epoch_actor_losses = []
            epoch_critic_losses = []
            for t_train in range(nb_train_step):
                cl, al = agent.train()
                epoch_actor_losses.append(al)
                epoch_critic_losses.append(cl)
                agent.update_target_network()


def main(args):
    env = gym.make(args.env_id)
    action_noise = None
    nb_action = env.action_space.shape[0]
    if args.noise_type == 'none':
        pass
    elif 'normal' in args.noise_type:
        _, stddev = args.noise_type.split('_')
        action_noise = NormalActionNoise(mu=np.zeros(nb_action), sigma=float(stddev) * np.ones(nb_action))
    elif 'ou' in args.noise_type:
        _, stddev = args.noise_type.split('_')
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_action), sigma=float(stddev) * np.ones(nb_action))

    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    # set random seed for all packages
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train(env=env, nb_epoch=args.nb_epoch, nb_epoch_cycle=args.nb_epoch_cycle, normalize_observations=args.normalize_observations,
          actor_lr=args.actor_lr, critic_lr=args.critic_lr, action_noise=action_noise,
          gamma=args.gamma, nb_train_step=args.nb_train_step, nb_rollout_step=args.nb_rollout_step,
          batch_size=args.batch_size, memory=memory)
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DDPG for deep reinforcement learning')
    parser.add_argument('--env_id', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=730, help='random seed (default: 730)')
    parser.add_argument('--normalize_observations', default=True, action='store_false', help='normalize observations (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma for discount (default: 0.99)')
    parser.add_argument('--noise_type', type=str, default='ou_0.2', help='noise type for action (default: Ornstein Uhlenbeck, sigma: 0.2)')
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='actor learning rate (default: 0.0001)')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='critic learning rate (default: 0.001)') 
    parser.add_argument('--nb_epoch', type=int, default=500, help='number of epoch for training (default: 500)')
    parser.add_argument('--nb_epoch_cycle', type=int, default=20, help='number of epoch cycle (default: 20)')
    parser.add_argument('--nb_train_step', type=int, default=50, help='number of train step (default: 50)')
    parser.add_argument('--nb_rollout_step', type=int, default=100, help='number of rollout_step (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 64)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
