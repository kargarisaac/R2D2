import os
import sys
import gym
import random
import numpy as np
import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import R2D2 
from memory import Memory, LocalBuffer
from tensorboardX import SummaryWriter

from config import initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr, sequence_length, local_mini_batch, env_config

from collections import deque

sys.path.append('/home/isaac/codes/deepdrive-zero/')
from deepdrive_zero.envs.env import Deepdrive2DEnv
from deepdrive_zero.constants import COMFORTABLE_STEERING_ACTIONS, \
    COMFORTABLE_ACTIONS


def get_action(state, target_net, epsilon, env, hidden):
    action, hidden = target_net.get_action(state, hidden)
    
    if np.random.rand() <= epsilon:
        return env.action_space.sample(), hidden
    else:
        return action, hidden

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


def main():
    # env = gym.make(env_name)
    env = Deepdrive2DEnv(is_intersection_map=True)
    env.configure_env(env_config)

    #========= set path variables ============
    algo_name = 'dd0_r2d2_single_'
    experiment_name = algo_name + str(datetime.datetime.today()).split(' ')[1].split('.')[0]
    yyyymmdd = datetime.datetime.today().strftime("%Y_%m_%d")
    experiment_name = os.path.join(yyyymmdd, experiment_name)
    base_checkpoint_path = f"./checkpoints/{experiment_name}/"
    log_dir = f"./logs/{experiment_name}/"
    
    if not os.path.exists(base_checkpoint_path):
        os.makedirs(base_checkpoint_path)

    #========= definitiions ==============
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = R2D2(num_inputs, num_actions, 128)
    target_net = R2D2(num_inputs, num_actions, 128)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    writer = SummaryWriter(log_dir)

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    running_score = 0
    epsilon = 1.0
    steps = 0
    loss = 0
    local_buffer = LocalBuffer(num_inputs)

    #=========== training loop =============
    for e in range(30000):
        done = False

        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)

        hidden = (torch.Tensor().new_zeros(1, 1, 128), torch.Tensor().new_zeros(1, 1, 128))

        while not done:
            steps += 1

            # epsilon greedy action selection and get hidden from target net
            action, new_hidden = get_action(state, target_net, epsilon, env, hidden)

            # apply action
            next_state, reward, done, info = env.step(action)

            next_state = torch.Tensor(next_state)

            mask = 0 if done else 1
            
            # store transition
            local_buffer.push(state, next_state, action, reward, mask, hidden)

            # update hidden
            hidden = new_hidden

            # local_buffer.memory contains a couple of sequences of data -> if we collected enough sequences: 
            if len(local_buffer.memory) == local_mini_batch:
                # get all collected sequences
                batch, lengths = local_buffer.sample()
                # calculate td_error
                td_error = R2D2.get_td_error(online_net, target_net, batch, lengths)
                # store data and calculated priority based on td_error in global memory(replay buffer)
                memory.push(td_error, batch, lengths)

            # add reward to score
            score += reward

            # update state for next step
            state = next_state
            
            # if did enough exploration and have enough data
            if steps > initial_exploration and len(memory) > batch_size:
                # update epsilon
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                # sample data (sequences) from memory and train model
                batch, indexes, lengths = memory.sample(batch_size)
                loss, td_error = R2D2.train_model(online_net, target_net, optimizer, batch, lengths)

                # update priorities for replay buffer
                memory.update_prior(indexes, td_error, lengths)

                # update target net
                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        # score = score if score == 500.0 else score + 1
        if running_score == 0:
            running_score = score
        else:
            running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, running_score, epsilon))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

            # save model and optimizer params
            torch.save(online_net.state_dict(), base_checkpoint_path + 'model.pt')
            torch.save(optimizer.state_dict(), base_checkpoint_path + 'optimizer.pt')

        if running_score > goal_score:
            break


if __name__=="__main__":
    main()
