'''
Source code: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py
'''

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils.utils import img_to_patch
from utils.masked_loss import MaskedCategorical


class PPOMemory:
    def __init__(self):
        self.states = []
        self.action_masks = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def generate_batches(self, batch_size=1):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        #np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        
        return self.states,\
               self.action_masks,\
               self.actions,\
               self.probs,\
               self.vals,\
               self.rewards,\
               self.dones,\
               batches

    def store_memory(self, state, action_mask, action, probs, vals, reward, done):
        self.states.append(state.detach().cpu())
        self.action_masks.append(action_mask.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.probs.append(probs.detach().cpu())
        self.vals.append(vals.detach().cpu())
        self.rewards.append(reward.detach().cpu())
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.action_masks = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr,
            fc1_dims=400, fc2_dims=400, chkpt_dir='tmp/ppo', model_name='gru', params_str=''):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, f'actor_ppo-{model_name}' + params_str)
        self.actor = nn.Sequential(
                nn.Linear(*input_dims + n_actions, fc1_dims),
                #nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action_mask):
        state = torch.cat((state, action_mask), dim=1)
        
        logits = self.actor(state)
        dists = MaskedCategorical(logits=logits, mask=action_mask.to(torch.bool))
        
        return dists

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr, fc1_dims=400, fc2_dims=400,
            chkpt_dir='tmp/ppo', model_name='gru', params_str=''):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, f'critic_ppo-{model_name}' + params_str)
        self.critic = nn.Sequential(
                nn.Linear(*input_dims + n_actions, fc1_dims),
                #nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action_mask):
        state = torch.cat((state, action_mask), dim=1)
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class PPOAgent:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10, seed=0, chkpt_dir='tmp/ppo', model_name='gru'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.n_actions = n_actions
        
        params_str = f"_lr{lr}_gamma{gamma}_gae{gae_lambda}_seed{seed}"

        self.actor = ActorNetwork(
            n_actions, input_dims, lr, chkpt_dir=chkpt_dir, model_name=model_name, params_str=params_str
        )
        self.critic = CriticNetwork(
            n_actions, input_dims, lr, chkpt_dir=chkpt_dir, model_name=model_name, params_str=params_str
        )
        self.memory = PPOMemory()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
       
    def remember(self, state, action_mask, action, probs, vals, reward, done):
        self.memory.store_memory(state, action_mask, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        #self.critic.load_checkpoint()

    def choose_action(self, observation, action_mask, determenistic=False):
        state = observation
        dists = self.actor(state, action_mask)
        values = self.critic(state, action_mask)
        
        if determenistic:
            actions = dists.mode
        else:    
            actions = dists.sample()
        
        probs = dists.log_prob(actions)
        values = torch.squeeze(values) 
        
        if determenistic:
            return actions, probs
                
        return actions, probs, values

    def learn(self):
        
        state_arr, action_mask_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()
        
        values = torch.stack(vals_arr, dim=0).to(self.actor.device)
        advantage = torch.zeros((len(reward_arr), vals_arr[0].size(0)), dtype=torch.float).to(self.actor.device)
        returns  = torch.zeros((len(reward_arr), vals_arr[0].size(0)), dtype=torch.float).to(self.actor.device)
        
        reward_arr = torch.stack(reward_arr, dim=0).to(self.actor.device)
        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            r_t = 0 
            for k in range(t, len(reward_arr) - 1):
                a_t += discount*(reward_arr[k] + self.gamma * values[k+1]*\
                        (1 - int(dones_arr[k])) - values[k])
                r_t += discount * reward_arr[k].float()
                
                discount *= self.gamma * self.gae_lambda
            #print(a_t.shape)    
            advantage[t] = a_t
            returns[t] = r_t
      
            
        #returns = (returns - returns.mean()) / (returns.std() + 1e-7)      
        #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-7)      
        num_timesteps = min(len(state_arr), self.n_actions - 1)
        bs = 16

        values = values.to(self.actor.device)
            
        for _ in range(1):
            total_loss = 0
            batch_inds = np.arange(num_timesteps)
            np.random.shuffle(batch_inds)

            for ind, batch in enumerate(batch_inds):
                if ind % bs == 0:
                    self.actor.optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()    
                        
                states = state_arr[batch].to(self.actor.device)
                action_masks = action_mask_arr[batch].to(self.actor.device)
                old_probs = old_prob_arr[batch].to(self.actor.device)
                actions = action_arr[batch].to(self.actor.device)

                dist = self.actor(states, action_masks)
                critic_value = self.critic(states, action_masks)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp() 
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                        1 + self.policy_clip) * advantage[batch]
                actor_loss = - torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value) ** 2
                critic_loss = critic_loss.mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().mean()
                loss.backward()

                total_loss += loss.item()

                if ((ind+1) % bs == 0) or ((ind + 1) == num_timesteps):
                #if ((ind + 1) == num_timesteps):
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                    self.actor.optimizer.step()
                    self.critic.optimizer.step()

        #self.memory.clear_memory()
        return total_loss
 
       