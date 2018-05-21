import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from model import Actor, Critic
from noise import *
from utils import *

criterion = nn.MSELoss()


class DDPG(object):
    def __init__(self, memory, nb_status, nb_actions, action_noise=None, 
                 gamma=0.99, tau=0.001, normalize_observations=True,
                 batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.),
                 actor_lr=1e-4, critic_lr=1e-3):
        self.nb_status = nb_status
        self.nb_actions = nb_actions
        self.observation_range = observation_range
        self.action_range = action_range
        self.normalize_observations = normalize_observations

        self.actor = Actor(self.nb_status, self.nb_actions)
        self.actor_target = Actor(self.nb_status, self.nb_actions)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(self.nb_status, self.nb_actions)
        self.critic_target = Critic(self.nb_status, self.nb_actions)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = memory
        self.action_noise = action_noise

        self.batch_size = batch_size
        self.discount = gamma
        self.tau = tau

        if self.normalize_observations:
            self.obs_rms = RunningMeanStd()
        else:
            self.obs_rms = None
    
    def initialization(self):
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
    
    def cuda(self, device):
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
    
    def store_transition(self, obs0, action, reward, obs1, terminal1):
        self.memory.append(obs0, action, reward, obs1, terminal1)
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))
    
    def update_target_network(self):
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
    
    def reset(self):
        if self.action_noise is not None:
            self.action_noise.reset()
    
    def policy(self, obs, apply_noise=True, compute_Q=True):
        obs = np.array([obs])
        action = self.actor(torch.from_numpy(obs)).numpy().squeeze(0)
        if compute_Q is True:
            q = self.critic([torch.from_numpy(obs), torch.from_numpy(action)]).data
        else:
            q = None
        if apply_noise is True:
            action += self.action_noise()
        
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q[0][0]
    
    def train(self):
        batch = self.memory.sample(batch_size=self.batch_size)

        next_q = self.critic_target([torch.from_numpy(batch['obs1']), 
                                     self.actor_target(torch.from_numpy(batch['obs1']))])
        target_q_batch = torch.from_numpy(batch['rewards']) + self.discount * torch.from_numpy(
            1 - batch['terminals1'].astype('float32')
        ) * next_q
        
        self.critic_optim.zero_grad()
        q_batch = self.critic([torch.from_numpy(batch['obs0']), torch.from_numpy(batch['actions'])])
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss = -self.critic([torch.from_numpy(batch['obs0']), self.actor(torch.from_numpy(batch['obs0']))]).mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.data[0], policy_loss.data[0]
