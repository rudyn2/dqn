"""
example_rllib_tune.py:

Example of how to use rllib and tune to execute an experiment.
"""
from ray import tune
import ray
from ray.rllib.agents import dqn, ppo

env = "CartPole-v1"
algorithm = "DQN"

config = dqn.DEFAULT_CONFIG.copy() if algorithm == "DQN" else ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 1
config["num_workers"] = 2
config["env"] = env
config["framework"] = "torch"


ray.init()
tune.run(
    algorithm,
    name="CartPole",
    stop=dict(episode_reward_mean=475),
    local_dir='results',
    config=config,
    checkpoint_at_end=True
)
