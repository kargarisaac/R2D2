import torch
import sys
sys.path.append('/home/kargarisaac/codes/deepdrive-zero/')
from deepdrive_zero.constants import COMFORTABLE_STEERING_ACTIONS, \
    COMFORTABLE_ACTIONS


resume = False


gamma = 0.99
batch_size = 64
lr = 1e-4
initial_exploration = 1e3
goal_score = 200
log_interval = 10
update_target = 2500
replay_memory_capacity = int(1e5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 32
burn_in_length = 16
eta = 0.9
local_mini_batch = 8
n_step = 5
over_lapping_length = 16

epsilon_scratch = 1.0
epsilon_resume = 1.0
epsilon_final = 0.01 #0.01
epsilon_step = 0.0001


env_config = dict(
        id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
        is_intersection_map=True,
        is_one_waypoint_map=False,
        expect_normalized_actions=True,
        expect_normalized_action_deltas=False,
        jerk_penalty_coeff=3.3e-6 * 1,
        gforce_penalty_coeff=0.0006 * 0,
        lane_penalty_coeff=0.02,  # 0.02,
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
        end_on_lane_violation=False
    )
