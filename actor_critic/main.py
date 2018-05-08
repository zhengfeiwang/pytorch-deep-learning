import argparse
import numpy as np
from itertools import count
from collections import namedtuple

import gym

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from policy_network import PolicyNetwork

parser = argparse.ArgumentParser(description='PyTorch actor-critic for CartPole-v0')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=730, help='random seed (default: 730)')
parser.add_argument('--render', action='store_true', default=False, help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, help='interval between training status logs (default: 10)')
args, unparsed = parser.parse_known_args()

eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedActon', ['log_prob', 'value'])
model = PolicyNetwork()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def train():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]
    

def main():
    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            model.rewards.append(reward)
            if done:
                break
        
        running_reward = running_reward * args.gamma + t * (1 - args.gamma)
        train()
        if i_episode % args.log_interval == 0:
            print('Episode {}\t last length: {:5d}\tAverage length: {:.2f}'.format(i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print('CartPole-v0 solved! Running reward is now {} and the last episode runs to {} time steps!'.format(running_reward, t))
            break


if __name__ == '__main__':
    main()
