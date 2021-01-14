import gym
import torch
from dqn import DQN


env = gym.make('CartPole-v1')
obs = env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
dqn = DQN(obs_dim, act_dim)
dqn.load_state_dict(torch.load('main_dqn.pth'))
dqn.eval()

for _ in range(1000):
    env.render()
    action = torch.argmax(dqn(obs)).item()
    _, _, done, _ = env.step(action)
    if done:
        print("Reset!")
        env.reset()

env.close()
