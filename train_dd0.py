import os
import sys
import gym
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import R2D2 
from memory import Memory, LocalBuffer
from tensorboardX import SummaryWriter

from config import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr, sequence_length, local_mini_batch

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

def state_to_partial_observability(state):
    state = state[[0, 2]]
    return state



env_config = dict(
        id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
        is_intersection_map=True,
        is_one_waypoint_map=False,
        expect_normalized_actions=True,
        expect_normalized_action_deltas=False,
        jerk_penalty_coeff=3.3e-6 * 0,
        gforce_penalty_coeff=0.0006 * 0,
        lane_penalty_coeff=0.01,  # 0.02,
        collision_penalty_coeff=4,
        speed_reward_coeff=0.50,
        gforce_threshold=None, #1.0,
        # jerk_threshold=150.0,  # 15g/s
        end_on_harmful_gs=False,
        incent_win=True,
        incent_yield_to_oncoming_traffic=True,
        constrain_controls=False,
        physics_steps_per_observation=6,
        contain_prev_actions_in_obs=True,
        discrete_actions=COMFORTABLE_ACTIONS,
        # dummy_accel_agent_indices=[1], #for opponent
        # dummy_random_scenario=True,
        end_on_lane_violation=True
    )


def main():
    # env = gym.make(env_name)
    env = Deepdrive2DEnv(is_intersection_map=False)
    env.configure_env(env_config)
    log_dir = 'logs/dd0/'

    env.seed(500)
    torch.manual_seed(500)

    # num_inputs = env.observation_space.shape[0]
    num_inputs = 2
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = R2D2(num_inputs, num_actions)
    target_net = R2D2(num_inputs, num_actions)
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
    local_buffer = LocalBuffer()

    for e in range(30000):
        done = False

        score = 0
        state = env.reset()
        state = state_to_partial_observability(state)
        state = torch.Tensor(state).to(device)

        hidden = (torch.Tensor().new_zeros(1, 1, 16), torch.Tensor().new_zeros(1, 1, 16))

        while not done:
            steps += 1

            action, new_hidden = get_action(state, target_net, epsilon, env, hidden)

            next_state, reward, done, _ = env.step(action)

            next_state = state_to_partial_observability(next_state)
            next_state = torch.Tensor(next_state)

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1

            local_buffer.push(state, next_state, action, reward, mask, hidden)
            hidden = new_hidden
            if len(local_buffer.memory) == local_mini_batch:
                batch, lengths = local_buffer.sample()
                td_error = R2D2.get_td_error(online_net, target_net, batch, lengths)
                memory.push(td_error, batch, lengths)

            score += reward
            state = next_state

            if steps > initial_exploration and len(memory) > batch_size:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                batch, indexes, lengths = memory.sample(batch_size)
                loss, td_error = R2D2.train_model(online_net, target_net, optimizer, batch, lengths)

                memory.update_prior(indexes, td_error, lengths)

                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        score = score if score == 500.0 else score + 1
        if running_score == 0:
            running_score = score
        else:
            running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, running_score, epsilon))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if running_score > goal_score:
            break


if __name__=="__main__":
    main()
