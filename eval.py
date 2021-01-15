import gym
import torch
from dqn import DQN


env = gym.make('CartPole-v1')

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
dqn = DQN(obs_dim, act_dim)
dqn.load_state_dict(torch.load('main_dqn.pth'))
dqn.eval()


# run until episode ends
episode_reward = 0
done = False
obs = env.reset()
while True:
    env.render()
    action = torch.argmax(dqn(obs)).item()
    obs, reward, done, info = env.step(action)
    episode_reward += reward

    if done:
        print(f"Episode reward: {episode_reward}")
        obs = env.reset()
        episode_reward = 0
