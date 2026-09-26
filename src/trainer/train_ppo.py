#import gym
import numpy as np
import torch

from utils.utils import img_to_patch

#from src.patch_selection.models.ppo import PPOAgent
#from src.patch_selection.models.environment import WaferMapGrid
from utils.eval import eval_policy

from datetime import datetime

#from utils import plot_learning_curve

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_one_epoch_ppo(agent, env):

    agent.actor.train()
    agent.critic.train()

    #best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    train_loader = env.train_loader
    delta_t = 32

    #epoch_loss = [0. for _ in range(patches.size(1))]
    score = 0
    total_loss = 0
    total_reward = 0

    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        patches = img_to_patch(X, patch_size=env.patch_size)
        done = False
        _, observation = env.reset(patches, y)

        if env.classifier.cell_type == "LSTM":
            observation = observation[0]

        t_step = 0
        total_loss_batch = 0

        discount = 1.0
        while not done:
            with torch.no_grad():
                action_mask = env.action_mask.clone()
                action, prob, val = agent.choose_action(observation, action_mask)
                y_pred, observation_, reward, done, info = env.step(action)

                if env.curr_time > env.start_policy:
                    total_reward += discount * reward.sum().float()
                    agent.remember(observation, action_mask, action, prob, val, reward, done)
                    discount *= agent.gamma

                observation = observation_


        total_loss_batch = agent.learn()
        agent.memory.clear_memory()

        print(f"Batch {i+1}/{len(train_loader)} | Last loss: {total_loss_batch:.4f}")

        total_loss += total_loss_batch * X.size(0)

    total_loss /= len(train_loader.dataset)
    total_reward /= len(train_loader.dataset)

    return agent, total_loss, total_reward


def training_loop_ppo(agent, env, n_epochs=100):
    best_score = -1e+10
    best_loss = 1e+10
    best_acc = 0

    train_loader = env.train_loader

    for epoch in range(0, n_epochs):
        agent, train_loss, train_score = train_one_epoch_ppo(agent, env)

        test_acc, _, test_score, _, _ = eval_policy(agent, env, train=False)

        if train_score > best_score:
            best_score = train_score
            agent.save_models()


        print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  #f'Valid loss: {valid_loss:.4f}\t'
                  #f'Train accuracy: {train_acc:.2f}\t'
                  f'Train reward: {train_score:.2f}\t'
                  f'Valid accuracy: {test_acc:.2f}\t'
                  f'Valid reward: {test_score:.2f}'
             )

    return agent
