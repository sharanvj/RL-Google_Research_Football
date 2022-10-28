'''
Minimal setup example with a random agent and using a configuration
'''
from gfootball.env import config
from gfootball.env import football_env


cfg_values = {
    'action_set': 'default',
    # 'custom_display_stats': None,
    # 'display_game_stats': True,
    # 'dump_full_episodes': False,
    # 'dump_scores': False,
    'players': ['agent:left_players=1'],
    'level': 'academy_empty_goal_close',
    # 'physics_steps_per_frame': 10,
    # 'render_resolution_x': 1280,
    # 'real_time': True,
    # 'render': True,
    # 'tracesdir': os.path.join(tempfile.gettempdir(), 'dumps'),
    # 'video_format': 'avi',
    # 'video_quality_level': 2,  # 0 - low, 1 - medium, 2 - high
    # 'write_video': False
}

cfg = config.Config(cfg_values)
env = football_env.FootballEnv(cfg)

env.render() # Only have to call this once to turn on rendering


env.reset()
steps = 0
ep_rew = 0
while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    steps += 1
    ep_rew += rew
    if done:
        print("Episode Steps: {} -- Episode Reward: {}".format(steps, ep_rew))
        steps = 0
        ep_rew = 0
        env.reset()