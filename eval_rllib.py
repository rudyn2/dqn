import ray
import gym
from ray.rllib.agents import dqn, ppo


ray.init()
env = "CartPole-v1"
algorithm = "PPO"
checkpoint_path = 'results/PPO/checkpoint_91/checkpoint-91'
config = dqn.DEFAULT_CONFIG.copy() if algorithm == "DQN" else ppo.DEFAULT_CONFIG.copy()
agent = dqn.DQNTrainer(config, env=env) if algorithm == "DQN" else ppo.PPOTrainer(config, env=env)
agent.restore(checkpoint_path)

env = gym.make(env)

# run until episode ends
episode_reward = 0
done = False
obs = env.reset()
while True:
    env.render()
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward

    if done:
        print(f"Episode reward: {episode_reward}")
        obs = env.reset()
        episode_reward = 0
