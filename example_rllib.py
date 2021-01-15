"""
example_rllib.py

Example of how to use rllib to find a solution of CartPole problem.
"""

from ray.rllib.agents import dqn, ppo
import ray

env = "CartPole-v1"
algorithm = "PPO"
checkpoint_path = f"results/{algorithm}"

ray.init()
config = dqn.DEFAULT_CONFIG.copy() if algorithm == "DQN" else ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"

agent = dqn.DQNTrainer(config, env=env) if algorithm == "DQN" else ppo.PPOTrainer(config, env=env)

N_ITER = 100
CHECKPOINT_FREQ = 10
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"
s_check = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"
for n in range(N_ITER):
    result = agent.train()
    s_print = s
    file_name = ""
    if n % CHECKPOINT_FREQ == 0:
        file_name = agent.save(checkpoint_path)
        s_print = s_check

    print(s_print.format(
        n + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
        file_name
    ))
