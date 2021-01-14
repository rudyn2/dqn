from ray.rllib.agents import dqn, ppo
import ray

env = "CartPole-v1"
checkpoint_path = "results"
algorithm = "DQN"

ray.init()
config = dqn.DEFAULT_CONFIG.copy() if algorithm == "DQN" else ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"

agent = dqn.DQNTrainer(config, env=env) if algorithm == "DQN" else ppo.PPOTrainer(config, env=env)

N_ITER = 40
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

for n in range(N_ITER):
    result = agent.train()
    file_name = agent.save(checkpoint_path)

    print(s.format(
        n + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
        file_name
    ))
