import os
import time
import random
import argparse
import numpy as np
import torch
import gym
from tensorboardX import SummaryWriter
from ddpg import DDPG
from evaluator import Evaluator


def train(nb_iterations, agent, env, evaluator):
    visualization = args.visualization
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    log = 0
    noise_level = args.noise_level * random.uniform(0, 1) / 2.
    time_stamp = time.time()

    while step <= nb_iterations:
        if observation is None:
            observation = env.reset()
            agent.reset(observation)

        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, exploration_noise=noise_level)

        observation, reward, done, info = env.step(action)
        if visualization:
            env.render()
        agent.observe(reward, observation, done)

        step += 1
        episode_steps += 1
        episode_reward += reward
        if done:
            if step > args.warmup:
                # validation
                if episode > 0 and episode % args.validate_interval == 0:
                    validation_reward = evaluator(env, agent.select_action, visualize=False)
                    print('[validation] episode #{}, reward={}'.format(episode, np.mean(validation_reward)))


            for i in range(episode_steps):
                if step > args.warmup:
                    log += 1
                    Q, value_loss = agent.update_policy()
                    writer.add_scalar('train/Q', Q.to(torch.device("cpu")).detach().numpy(), log)
                    writer.add_scalar('train/critic loss', value_loss.to(torch.device("cpu")).detach().numpy(), log)

            writer.add_scalar('train/train_reward', episode_reward, episode)

            # log
            train_time = time.time() - time_stamp
            time_stamp = time.time()
            print('episode#{}: reward={}, steps={}, noise_level={:.2f}, time={:.2f}'.format(
                    episode, episode_reward, episode_steps, noise_level, train_time
            ))

            observation = None
            episode_steps = 0
            episode_reward = 0
            episode += 1
            noise_level = args.noise_level * random.uniform(0, 1) / 2.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG implemented by PyTorch')
    parser.add_argument('--env', default='CartPole-v0', type=str, help='OpenAI Gym environment')
    parser.add_argument('--discrete', dest='discrete', action='store_true')
    parser.add_argument('--discount', default=0.99, type=float, help='bellman discount')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--memory_size', default=1000000, type=int, help='memory size')

    parser.add_argument('--hidden1', default=400, type=int, help='number of first fully connected layer')
    parser.add_argument('--hidden2', default=300, type=int, help='number of second fully connected layer')
    parser.add_argument('--actor_lr', default=1e-4, type=float, help='actor learning rate')
    parser.add_argument('--critic_lr', default=1e-3, type=float, help='critic learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='L2 weight decay')

    parser.add_argument('--iterations', default=2000000, type=int, help='iterations during training')
    parser.add_argument('--warmup', default=1000, type=int, help='timestep without training to fill the replay buffer')
    parser.add_argument('--noise_level', default=1, type=float, help='noise level added to the action')
    parser.add_argument('--validate_interval', default=10, type=int, help='how many episodes to validate')
    parser.add_argument('--save_interval', default=100, type=int, help='how many episodes to save model')

    parser.add_argument('--validation_episodes', default=1, type=int, help='number of episodes during validation')
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--visualization', dest='visualization', action='store_true')
    parser.add_argument('--cuda', dest='cuda', action='store_true')

    parser.add_argument('--seed', default=-1, type=int, help='random seed')

    args = parser.parse_args()

    writer = SummaryWriter(os.path.join(args.output, args.env))

    if args.discrete:
        env = gym.make(args.env)
        env = env.unwrapped
    else:
        env = gym.make(args.env)

    # set random seed
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        env.seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    # states and actions space
    nb_states = env.observation_space.shape[0]
    if args.discrete:
        nb_actions = env.action_space.n
    else:
        nb_actions = env.action_space.shape[0]

    evaluator = Evaluator(args)

    agent = DDPG(nb_states, nb_actions, args)
    train(args.iterations, agent, env, evaluator)
