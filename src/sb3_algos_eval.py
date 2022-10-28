'''
SB3 discrete-action algorithms test
'''
from gc import callbacks
from tabnanny import verbose
import gfootball.env as football_env

from stable_baselines3 import A2C, DQN, PPO
from sb3_contrib import ARS, QRDQN, TRPO, MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import tensorboard

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

timesteps = 100000

i_scen = 2

tensorboard_log = ("../sb3_test_logs/" + academy_scenarios[i_scen])

env = football_env.create_environment(
    env_name=academy_scenarios[i_scen], 
    representation='simple115v2',
    render=True)
    # logdir='../simple_vid/',
    # write_full_episode_dumps=True,
    # write_goal_dumps=False,
    # write_video=True)

# a2c = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
# a2c.learn(total_timesteps=timesteps)
# a2c.save("../a2c_acad3_test")
# env.reset()

# ars = ARS("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
# ars.learn(total_timesteps=timesteps)
# ars.save("../ars_acad3_test")
# env.reset()

# trpo = TRPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
# trpo.learn(total_timesteps=timesteps)
# trpo.save("../trpo_acad3_test")
# env.reset()

# dqn = DQN("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
# dqn.learn(total_timesteps=timesteps)
# dqn.save("../dqn_acad3_test")
# env.reset()

# ppo = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
# ppo.learn(total_timesteps=timesteps)
# ppo.save("../ppo_acad3_test")
# env.reset()

# qrdqn = QRDQN("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
# qrdqn.learn(total_timesteps=timesteps)
# qrdqn.save("../qrdqn_acad3_test")

# env.close()










model = A2C.load("../a2c_acad3_test.zip", env=env)
# model = ARS.load("../ars_acad4_test.zip", env=env)
# model = TRPO.load("../trpo_acad4_test.zip", env=env)
# model = DQN.load("../dqn_acad4_test.zip", env=env)
# model = PPO.load("../ppo_acad4_test.zip", env=env)
# model = QRDQN.load("../qrdqn_acad4_test.zip", env=env)

steps = 0
ep_rew = 0
# env.render()
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rew, done, info = env.step(action)
    steps += 1
    ep_rew += rew
    if done:
        print("Episode Steps: {} -- Episode Reward: {}".format(steps, ep_rew))
        steps = 0
        ep_rew = 0
        obs = env.reset()
        # break
env.close()
    