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

from config import initial_exploration, batch_size, update_target, log_interval, device, replay_memory_capacity, \
 lr, sequence_length, local_mini_batch, env_config, resume, epsilon_scratch, epsilon_resume, epsilon_final, epsilon_step

from collections import deque

sys.path.append('/home/kargarisaac/codes/deepdrive-zero/')
from deepdrive_zero.envs.env import Deepdrive2DEnv
from deepdrive_zero.constants import COMFORTABLE_STEERING_ACTIONS, \
    COMFORTABLE_ACTIONS


from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.vector_env import VectorEnv


def get_action(state, target_net, epsilon, env, hidden):
    # epsilon greedy action selection
    action, hidden = target_net.get_action(state, hidden)
    
    if np.random.rand() <= epsilon:
        return env.action_space.sample(), hidden
    else:
        return action, hidden

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


def make_env(env_cls, num_envs=1, asynchronous=True, wrappers=None, env_config=None, **kwargs):
    """Create a vectorized environment from multiple copies of an environment,
    from its id
    Parameters
    ----------
    id : str
        The environment ID. This must be a valid ID from the registry.
    num_envs : int
        Number of copies of the environment.
    asynchronous : bool (default: `True`)
        If `True`, wraps the environments in an `AsyncVectorEnv` (which uses
        `multiprocessing` to run the environments in parallel). If `False`,
        wraps the environments in a `SyncVectorEnv`.

    wrappers : Callable or Iterable of Callables (default: `None`)
        If not `None`, then apply the wrappers to each internal
        environment during creation.
    Returns
    -------
    env : `gym.vector.VectorEnv` instance
        The vectorized environment.
    Example
    -------
    # >>> import gym
    # >>> env = gym.vector.make('CartPole-v1', 3)
    # >>> env.reset()
    array([[-0.04456399,  0.04653909,  0.01326909, -0.02099827],
           [ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
           [ 0.03468829,  0.01500225,  0.01230312,  0.01825218]],
          dtype=float32)
    """
    from gym.envs import make as make_
    def _make_env():
        #         env = make_(id, **kwargs)
        env = env_cls(is_intersection_map = True)  # for self-play to have 2 learning agents
        env.configure_env(env_config)

        if wrappers is not None:
            if callable(wrappers):
                env = wrappers(env)
            elif isinstance(wrappers, Iterable) and all([callable(w) for w in wrappers]):
                for wrapper in wrappers:
                    env = wrapper(env)
            else:
                raise NotImplementedError
        return env

    env_fns = [_make_env for _ in range(num_envs)]
    return AsyncVectorEnv(env_fns) if asynchronous else SyncVectorEnv(env_fns)



def main():

    n_envs = 100
    
    torch.manual_seed(500)
    torch.set_num_threads(10)

    hidden_size = 128
    
    env = make_env(Deepdrive2DEnv, n_envs, asynchronous=True, env_config=env_config)
    env.seed(500)

    #========= set path variables ============
    algo_name = 'dd0_r2d2_parallel_selfplay_'
    experiment_name = algo_name + str(datetime.datetime.today()).split(' ')[1].split('.')[0]
    yyyymmdd = datetime.datetime.today().strftime("%Y_%m_%d")
    experiment_name = os.path.join(yyyymmdd, experiment_name)
    base_checkpoint_path = f"./checkpoints/{experiment_name}/"
    log_dir = f"./logs/{experiment_name}/"
    
    if not os.path.exists(base_checkpoint_path):
        os.makedirs(base_checkpoint_path)

    #========= definitiions ==============
    
    num_inputs = env.observation_space.shape[1]
    num_actions = env.action_space[0].n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = R2D2(num_inputs, num_actions, hidden_size, n_envs)
    target_net = R2D2(num_inputs, num_actions, hidden_size, n_envs)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    if resume:
        checkpoint_path = 'checkpoints/2020_06_16/dd0_r2d2_parallel_selfplay_17:32:58/'
        online_net.load_state_dict(torch.load(checkpoint_path + 'model.pt'))
        optimizer.load_state_dict(torch.load(checkpoint_path + 'optimizer.pt'))

    update_target_model(online_net, target_net)
    
    writer = SummaryWriter(log_dir)

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    
    if resume:
        epsilon = epsilon_resume
    else:
        epsilon = epsilon_scratch
    
    local_buffer = LocalBuffer(num_inputs, hidden_size, n_envs)

    #=========== training loop =============
    
    score = np.zeros(n_envs)
    score_to_show = [] # a list to store rewards of done trajectories and take mean of them to show in tensorboard
    state = env.reset()
    hidden = (torch.Tensor().new_zeros(1, n_envs, hidden_size), torch.Tensor().new_zeros(1, n_envs, hidden_size)) #[number of layers, n_envs, hidden_size]

    for step in range(100000):
        
        state = torch.Tensor(state).to(device)
        
        # epsilon greedy action selection and get hidden from target net
        action, new_hidden = get_action(state, target_net, epsilon, env, hidden)
        action = np.array(action)

        # apply action
        next_state, reward, done, info = env.step(action)

        next_state = torch.Tensor(next_state)

        mask = np.ones(n_envs) - done
        
        # store transitions for all envs
        local_buffer.push(state, next_state, action, reward, mask, hidden)

        # update hidden
        hidden = new_hidden

        # local_buffer.memory contains a couple of sequences of data -> if we collected enough sequences: 
        if len(local_buffer.memory) >= local_mini_batch:
            # get all collected sequences
            batch, lengths = local_buffer.sample()
            # calculate td_error
            td_error = R2D2.get_td_error(online_net, target_net, batch, lengths)
            # store data and calculated priority based on td_error in global memory(replay buffer)
            memory.push(td_error, batch, lengths)

        # add reward to score
        score += reward
        for n in range(n_envs):
            if done[n]:
                hidden = list(hidden)
                hidden[0][:, n, :] = torch.Tensor().new_zeros(1, 1, hidden_size)
                hidden[1][:, n, :] = torch.Tensor().new_zeros(1, 1, hidden_size)
                hidden = tuple(hidden)
                score_to_show.append(score[n])
                score[n] = 0
                
        # update state for next step
        state = next_state
        
        # if did enough exploration and have enough data
        if step > initial_exploration and len(memory) > batch_size:
            # update epsilon
            epsilon -= epsilon_step
            epsilon = max(epsilon, epsilon_final)

            # sample data (sequences) from memory and train model
            batch, indexes, lengths = memory.sample(batch_size)
            loss, td_error = R2D2.train_model(online_net, target_net, optimizer, batch, lengths)

            # update priorities for replay buffer
            memory.update_prior(indexes, td_error, lengths)

            # update target net
            if step % update_target == 0:
                update_target_model(online_net, target_net)

        
        if step % log_interval == 0:
            print('{} step | score: {:.2f} | epsilon: {:.2f}'.format(
                step, np.mean(score_to_show), epsilon))
            writer.add_scalar('log/score', float(np.mean(score_to_show)), step)
            writer.add_scalar('log/epsilon', float(epsilon), step)

            score_to_show = [] # reset the list for next log step

            # save model and optimizer params
            torch.save(online_net.state_dict(), base_checkpoint_path + 'model.pt')
            torch.save(optimizer.state_dict(), base_checkpoint_path + 'optimizer.pt')



def evaluate():
    env = Deepdrive2DEnv(is_intersection_map=True)
    env.configure_env(env_config)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    hidden_size = 128

    online_net = R2D2(num_inputs, num_actions, hidden_size)

    online_net.load_state_dict(torch.load('/home/kargarisaac/codes/R2D2/checkpoints/2020_06_17/dd0_r2d2_parallel_selfplay_04:27:57/model.pt'))

    online_net.to(device)
    online_net.eval()
    
    with torch.no_grad():

    	for _ in range(5):
	    
	        score = 0
	        state = env.reset()

	        hidden = (torch.Tensor().new_zeros(1, 1, hidden_size), torch.Tensor().new_zeros(1, 1, hidden_size))

	        while True:
	            
	            state = torch.Tensor(state).to(device)

	            action, hidden = online_net.get_action_eval(state, hidden)

	            # apply action
	            state, reward, done, info = env.step(action)

	            # add reward to score
	            score += reward

	            env.render()

	            if done:
	                # state = env.reset()
	                break

	        print(f'score:{score}')



if __name__=="__main__":
    # main()
    evaluate()
