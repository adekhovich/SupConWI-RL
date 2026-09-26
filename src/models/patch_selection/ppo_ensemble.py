'''
Source code: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py

NOTE: unlike the single-model PPOAgent (ppo.py), this ensemble agent is inference-only,
matching the design of the reference implementation for the predecessor paper
(https://github.com/adekhovich/sequential_wafer_inspection, REINFORCEAgentEnsemble):
train `num_models` independent single-agent policies via the ordinary single-model
pipeline (each with a different --seed), then load and combine their decisions here at
evaluation time. There is no joint ensemble-training step (see parser.py's --num_models
help text: "ONLY FOR EVALUATION").
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


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr,
            fc1_dims=400, fc2_dims=400, chkpt_dir='tmp/ppo', model_name='gru', params_str='', seed=0):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, f'actor_ppo-{model_name}' + params_str + f"_seed{seed}")
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
        #dists = MaskedCategorical(logits=logits, mask=action_mask.to(torch.bool))

        return logits

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr, fc1_dims=400, fc2_dims=400,
            chkpt_dir='tmp/ppo', model_name='gru', params_str='', seed=0):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, f'critic_ppo-{model_name}' + params_str + f"_seed{seed}")
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

class PPOAgentEnsemble:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10, num_models=5, ensemble_type='min-entropy', seeds=[0, 1, 2, 3, 4], chkpt_dir='tmp/ppo', model_name='gru'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.n_actions = n_actions
        self.ensemble_type = ensemble_type
        self.num_models = num_models

        params_str = f"_lr{lr}_gamma{gamma}_gae{gae_lambda}"

        # Each of these is an independently pretrained single-model actor/critic
        # (see PPOAgent in ppo.py), identified by its own seed. They are loaded via
        # load_models() below and combined only at evaluation time in choose_action().
        self.actors = [ActorNetwork(n_actions, input_dims, lr, chkpt_dir=chkpt_dir, model_name=model_name, params_str=params_str, seed=seed) for seed in seeds]
        self.critics = [CriticNetwork(n_actions, input_dims, lr, chkpt_dir=chkpt_dir, model_name=model_name, params_str=params_str, seed=seed) for seed in seeds]

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def save_models(self):
        print('... saving models ...')

        for i in range(self.num_models):
            self.actors[i].save_checkpoint()
            self.critics[i].save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        for i in range(self.num_models):
            self.actors[i].load_checkpoint()
        #for i in range(self.num_models):
        #    self.critics[i].load_checkpoint()

    def choose_action(self, observation, action_mask, determenistic=False):
        logits_avg = 0
        probs_avg = 0
        entropy = torch.zeros((observation[0].size(0), self.num_models)).to(self.device)
        probs_arr = torch.zeros((observation[0].size(0), self.num_models, self.n_actions)).to(self.device)

        for i in range(self.num_models):
            logits = self.actors[i](observation[i], action_mask)
            logits_avg += logits
            p = F.softmax(logits, dim=-1)
            probs_avg += p

            if self.ensemble_type == 'min-entropy':
                probs_arr[:, i, :] = p.clone()
                p[~action_mask.to(torch.bool)] = 0
                p = p / p.sum(dim=-1, keepdim=True)
                entropy[:, i] =  - ( p * torch.log(p)).sum(dim=1)#

        logits_avg /= self.num_models
        probs_avg /= self.num_models
        if self.ensemble_type == 'min-entropy':
            pi_idx = entropy.min(dim=-1)[1]
            probs_avg = torch.stack([probs_arr[i, pi_i, :] for i, pi_i in enumerate(pi_idx)], dim=0)

        if determenistic:
            probs = (probs_avg * action_mask) / (probs_avg * action_mask).sum(dim=-1, keepdim=True)
            actions = probs.argmax(dim=-1)
        else:
            dists = MaskedCategorical(probs=probs_avg, mask=action_mask.to(torch.bool))
            actions = dists.sample()
            probs = dists.log_prob(actions)

        return actions, probs
