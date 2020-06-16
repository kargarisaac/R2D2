import random
from collections import namedtuple, deque
from config import sequence_length, burn_in_length, eta, n_step, gamma, over_lapping_length
import torch
import numpy as np

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask', 'step', 'rnn_state'))


class LocalBuffer(object):
    def __init__(self, observation_size, hidden_size, n_envs=1):
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.n_envs = n_envs

        if n_envs > 1:
            self.n_step_memory = [[] for n in range(n_envs)]
        else:
            self.n_step_memory = []
        self.local_memory = []
        self.memory = []
        self.over_lapping_from_prev = []

    def push(self, state, next_state, action, reward, mask, rnn_state):
        """
        push new data into n_step_memory and if it is full or if the episode is doen -> 
        creqate one sample from this n_step data and push it into local_memory
        """
        if self.n_envs > 1:
            for n in range(self.n_envs):
                self.n_step_memory[n].append([state[n], next_state[n], action[n], reward[n], mask[n], tuple([rnn_state[0][:, n:n+1, :], rnn_state[1][:, n:n+1, :]])])
        else:
            self.n_step_memory.append([state, next_state, action, reward, mask, rnn_state])

        # if we collected s_step transitions or if the episode is done -> creqate one sample from this n_step data and push it into local memory 
        if self.n_envs > 1:
            for n in range(self.n_envs):
                if len(self.n_step_memory[n]) == n_step or mask[n] == 0.:
                    [state, _, action, _, _, rnn_state] = self.n_step_memory[n][0]
                    [_, next_state, _, _, mask2, _] = self.n_step_memory[n][-1]

                    sum_reward = 0
                    for t in reversed(range(len(self.n_step_memory[n]))):
                        [_, _, _, reward, _, _] = self.n_step_memory[n][t]
                        sum_reward += reward + gamma * sum_reward
                    reward = sum_reward
                    step = len(self.n_step_memory[n])
                    self.push_local_memory(state, next_state, action, reward, mask2, step, rnn_state)
                    self.n_step_memory[n] = []

        else:
            if len(self.n_step_memory)  == n_step or mask == 0:
                [state, _, action, _, _, rnn_state] = self.n_step_memory[0]
                [_, next_state, _, _, mask, _] = self.n_step_memory[-1]

                sum_reward = 0
                for t in reversed(range(len(self.n_step_memory))):
                    [_, _, _, reward, _, _] = self.n_step_memory[t]
                    sum_reward += reward + gamma * sum_reward
                reward = sum_reward
                step = len(self.n_step_memory)
                self.push_local_memory(state, next_state, action, reward, mask, step, rnn_state)
                self.n_step_memory = []

    
    def push_local_memory(self, state, next_state, action, reward, mask, step, rnn_state):
        """
        store data calculated from n_step to local_memory and add local_memory+over_lap_from_prev to memory if we have sequence_length data .
        So self.memory will contain sequences of Transition namedtupples
        """
        self.local_memory.append(Transition(state, next_state, action, reward, mask, step, torch.stack(rnn_state).view(2, -1)))
        if (len(self.local_memory) + len(self.over_lapping_from_prev)) == sequence_length or mask == 0:
            self.local_memory = self.over_lapping_from_prev + self.local_memory
            length = len(self.local_memory)
            while len(self.local_memory) < sequence_length:
                self.local_memory.append(Transition(
                    torch.Tensor(torch.zeros(self.observation_size)),
                    torch.Tensor(torch.zeros(self.observation_size)),
                    0,
                    0,
                    0,
                    0,
                    torch.zeros([2, 1, self.hidden_size]).view(2, -1)
                ))
            self.memory.append([self.local_memory, length])
            if mask == 0:
                self.over_lapping_from_prev = []
            else:
                self.over_lapping_from_prev = self.local_memory[len(self.local_memory) - over_lapping_length:]
            self.local_memory = []
    
    def sample(self):
        episodes = self.memory
        batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_step, batch_rnn_state = [], [], [], [], [], [], []
        lengths = []
        for episode, length in episodes:
            batch = Transition(*zip(*episode))

            batch_state.append(torch.stack(list(batch.state)))
            batch_next_state.append(torch.stack(list(batch.next_state)))
            batch_action.append(torch.Tensor(list(batch.action)))
            batch_reward.append(torch.Tensor(list(batch.reward)))
            batch_mask.append(torch.Tensor(list(batch.mask)))
            batch_step.append(torch.Tensor(list(batch.step)))
            batch_rnn_state.append(torch.stack(list(batch.rnn_state)))

            lengths.append(length)
        self.memory = []
        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_step, batch_rnn_state), lengths



class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.memory_probability = deque(maxlen=capacity)

    def td_error_to_prior(self, td_error, lengths):
        """
        calculate priority based on td_error
        """
        abs_td_error_sum  = td_error.abs().sum(dim=1, keepdim=True).view(-1).detach().numpy()
        lengths_burn = np.array([max(1, length - burn_in_length) for length in lengths])

        prior_max = td_error.abs().max(dim=1, keepdim=True)[0].view(-1).detach().numpy()

        prior_mean = abs_td_error_sum / (lengths_burn) #+ 1e-10)
        prior = eta * prior_max + (1 - eta) * prior_mean
        return prior

    def push(self, td_error, batch, lengths):
        """
        store data and priority
        """
        # batch.state[local_mini_batch, sequence_length, item]
        prior = self.td_error_to_prior(td_error, lengths)
        
        for i in range(len(batch)):
            self.memory.append([Transition(batch.state[i], batch.next_state[i], batch.action[i], batch.reward[i], batch.mask[i], batch.step[i], batch.rnn_state[i]), lengths[i]])
            self.memory_probability.append(prior[i])
        
    def sample(self, batch_size):
        """
        sample data from memory based on priority
        """
        probability = np.array(self.memory_probability)
        probability = probability / probability.sum()

        indexes = np.random.choice(range(len(self.memory_probability)), batch_size, p=probability)
        # indexes = np.random.choice(range(len(self.memory_probability)), batch_size)
        episodes = [self.memory[idx][0] for idx in indexes]
        lengths = [self.memory[idx][1] for idx in indexes]

        batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_step, batch_rnn_state = [], [], [], [], [], [], []
        for episode in episodes:
            batch_state.append(episode.state)
            batch_next_state.append(episode.next_state)
            batch_action.append(episode.action)
            batch_reward.append(episode.reward)
            batch_mask.append(episode.mask)
            batch_step.append(episode.step)
            batch_rnn_state.append(episode.rnn_state)
        
        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_step, batch_rnn_state), indexes, lengths
    
    def update_prior(self, indexes, td_error, lengths):
        prior = self.td_error_to_prior(td_error, lengths)
        priors_idx = 0
        for idx in indexes:
            self.memory_probability[idx] = prior[priors_idx]
            priors_idx += 1
    
    def __len__(self):
        return len(self.memory)
