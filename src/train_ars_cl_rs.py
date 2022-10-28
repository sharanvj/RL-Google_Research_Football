'''
ARS with training on curriculum defined by first few academy problems, and "easy" policy transfer
'''
import gfootball.env as football_env
from sb3_contrib import ARS, QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import tensorboard
import os
import gym
import numpy as np
import math

TOTAL_STEPS = 0

def get_shaped_rewards(observation, reward, n_steps, action):
    old_timestep = 0
    old_pos = None
    timestep = n_steps
    const = 0.003
    reward_shaped = 0
    obs = observation
    coordinates_left_team = obs[0: 22]
    ball_pos = obs[88: 91]
    ball_ownership = obs[94:97]
    active_player_map = obs[97:108]
    player_coordinates = [(coordinates_left_team[0], coordinates_left_team[1]),
                          (coordinates_left_team[2], coordinates_left_team[3])]
    result = np.where(active_player_map == 1)
    # print('\ncoordinates_left_team: ', player_coordinates[result[0][0]])
    # print('\nball position        : ', ball_pos)
    # print('\nball ownership       : ', ball_ownership)
    # print('\nactive player map    : ', result[0][0])
    active_player = player_coordinates[result[0][0]]
    # dist_to_goal1 = math.sqrt((active_player[0] - 1) ** 2 + (
    #         active_player[1] - 0.044) ** 2)
    # print(dist_to_goal1)
    if old_pos is None:
        old_pos = active_player

    if ball_ownership[0] == 0:
        dist_to_ball = math.sqrt((ball_pos[0] - active_player[0]) ** 2 + (ball_pos[1] - active_player[1]) ** 2)
        reward_shaped = ((0.25 * const) / (dist_to_ball + const)) - 0.1
    else:
        if timestep - old_timestep > 0:
            old_timestep = timestep
            if active_player[0] - old_pos[0] == 0 and active_player[1] - old_pos[1] == 0:  # check if position changed?
                reward_shaped = -0.08 * (timestep - old_timestep)
                old_pos = active_player

            else:
                dist_to_goal = math.sqrt((active_player[0] - 1) ** 2 + (
                        active_player[1] - 0.044) ** 2)  # check distance to goal from an activated player with ball

                reward_shaped = ((0.2 * const) / (dist_to_goal + const)) - 0.1
                if dist_to_goal < 0.35:
                    if action == 12:
                        reward_shaped = 0.15
                else:
                    if action == 12:
                        reward_shaped = -0.15

    # print('\ndistance to ball     : ', dist_to_ball)
    # print('\ndistance to goal     : ', dist_to_goal)
    # print('\nreward               : ', reward_)

    return reward_shaped


class RSWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steps = 0
        self.env = env
        
    def step(self, action):
        self.steps += 1
        next_state, reward, done, info = self.env.step(action)
        shaped_reward = get_shaped_rewards(next_state, reward, self.steps, action)
        return next_state, shaped_reward, done, info

    def reset(self):
        self.steps = 0
        self.env.reset()
        return


academy_scenarios = [
    'academy_empty_goal_close',
    'academy_empty_goal',
    'academy_run_to_score',
    'academy_run_to_score_with_keeper',
    'academy_pass_and_shoot_with_keeper',
    'academy_run_pass_and_shoot_with_keeper',
    'academy_3_vs_1_with_keeper',
    'academy_corner',
    'academy_counterattack_easy',
    'academy_counterattack_hard',
    'academy_single_goal_versus_lazy',
]

total_timesteps = 100000
tensorboard_log = ("../ars_cl_rs_training_logs/")
model_name = "ars_cl_rs"

for i_scen in range(4):

    print("=== New Training Scenario: {} ===".format(academy_scenarios[i_scen]))

    env = football_env.create_environment(
        env_name=academy_scenarios[i_scen], 
        representation='simple115v2',
        render=False)
    #     logdir='../easy_ars_{}/'.format(academy_scenarios[i_scen]),
    #     write_full_episode_dumps=True,
    # # # write_goal_dumps=False,
    #     write_video=True)

    env = RSWrapper(env)

    # Check if model exists and load/create
    if os.path.isfile("../{}.zip".format(model_name)):
        print("Using stored model [{}]!".format(model_name))
        model = ARS.load("../{}.zip".format(model_name), env=env)
        reset_num = True
    else:
        model = ARS("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
        reset_num = False
    
    model.learn(total_timesteps=total_timesteps, tb_log_name=academy_scenarios[i_scen], reset_num_timesteps=reset_num)

    model.save("../{}".format(model_name))

    env.close()
