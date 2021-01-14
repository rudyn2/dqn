from ray.rllib.agents.dqn import DQNTrainer
import yaml
import ray
import gym


with open('cart-pole-dqn-rllib.yaml', 'r') as f:
    config = yaml.load(f)

ray.init()
checkpoint_path = 'results/DQN-420-stop/DQN_CartPole-v1_1_n_step=10_2021-01-14_15-23-53_ebwzqvn/checkpoint_67/checkpoint-67'
agent = DQNTrainer(config=config['config'], env=config["env"])
agent.restore(checkpoint_path)

env = gym.make(config['env'])

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
