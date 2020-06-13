import torch
import sys
sys.path.append('/home/isaac/codes/deepdrive-zero/')
from deepdrive_zero.constants import COMFORTABLE_STEERING_ACTIONS, \
    COMFORTABLE_ACTIONS


# env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 1000
goal_score = 200
log_interval = 10
update_target = 100
replay_memory_capacity = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 32
burn_in_length = 4
eta = 0.9
local_mini_batch = 8
n_step = 2
over_lapping_length = 16


# gamma = 0.99
# batch_size = 64
# lr = 0.001
# initial_exploration = 1000
# goal_score = 100
# log_interval = 10
# update_target = 100
# replay_memory_capacity = 10000
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sequence_length = 32
# burn_in_length = 4
# eta = 0.9
# local_mini_batch = 8
# n_step = 5
# over_lapping_length = 16

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
        end_on_lane_violation=False
    )
