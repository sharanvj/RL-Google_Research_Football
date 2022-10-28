'''
ARS with training on curriculum defined by first few academy problems, and "easy" policy transfer
'''
import gfootball.env as football_env
from sb3_contrib import ARS
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import tensorboard
import os 

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

class ConvergenceCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(ConvergenceCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        print(self.logger.__dict__)
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


total_timesteps = 25000
conv_cb = ConvergenceCallback()
tensorboard_log = ("../easy_ars_training_logs/")

for i_scen in range(3,4):

    print("=== New Training Scenario: {} ===".format(academy_scenarios[i_scen]))

    env = football_env.create_environment(
        env_name=academy_scenarios[i_scen], 
        representation='simple115v2',
        render=False)

    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

    # Check if model exists and load/create
    if os.path.isfile("../easy_ars.zip"):
        print("Using stored easy_ars model!")
        model = ARS.load("../easy_ars.zip", env=env)
    else:
        print("Creating initial easy_ars model...")
        model = ARS("LinearPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, tb_log_name=academy_scenarios[i_scen])

    model.save("../easy_ars")

    env.close()



# go through algos, put notes/thoughts on why some are working/some aren't, might be complex behavior, etc
# videos for each algo