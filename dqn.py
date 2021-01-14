import torch
import gym
import random
import numpy as np


class Experience:
    def __init__(self, state, next_state, action, reward, done):
        self.state = state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.done = done


class ExperienceReplay:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []

    def __len__(self):
        return len(self.data)

    def put(self, t: Experience):
        if len(self.data) >= self.capacity:
            self.data = self.data[1:]
        self.data.append(t)

    def sample(self, size: int):
        if size < len(self.data):
            return random.sample(self.data, size)
        return self.data


class DQN(torch.nn.Module):
    """
    Q-Value function approximation
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim * 2)
        self.fc2 = torch.nn.Linear(input_dim * 2, output_dim)

    def forward(self, state):
        """
        Returns Q(state, *)
        """
        x = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), dim=0)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


class DQNTrainer:
    def __init__(self, main_dqn: DQN, target_dqn: DQN, env: gym.Env):
        self.env = env
        self.main_dqn = main_dqn
        self.target_dqn = target_dqn
        self.e_greedy = 1.0
        self.e_min = 0.01
        self.e_decay = 0.995
        self.gamma = 0.618
        self.batch_size = 64 * 2
        self.steps_to_update_target_model = 500
        self.steps_to_decrease_e_greedy = 200
        self.buffer = ExperienceReplay(5000)
        self.optimizer = torch.optim.Adam(self.main_dqn.parameters(), lr=0.001)

    def train(self, episodes: int):
        steps_to_update_target_model = 0
        steps_to_decrease_e_greedy = 0

        for episode in range(episodes + self.batch_size):
            state = self.env.reset()
            losses = []
            # episode update
            steps = 0
            accum_reward = 0
            while True:
                steps_to_update_target_model += 1
                steps_to_decrease_e_greedy += 1

                # epsilon greedy strategy
                if random.random() < self.e_greedy:
                    action = self.env.action_space.sample()
                else:
                    action = torch.argmax(self.main_dqn(state)).item()

                # perform action
                next_state, r, done, _ = self.env.step(action)
                accum_reward += r
                new_trans = Experience(state, next_state, action, r, done)
                self.buffer.put(new_trans)

                if len(self.buffer) < self.batch_size:
                    break

                batch_trans = self.buffer.sample(self.batch_size)
                loss = self.replay(batch_trans)
                losses.append(loss.item())
                self.update(loss)

                if steps_to_update_target_model >= self.steps_to_update_target_model:
                    print('Copying main network weights to the target network weights')
                    self.target_dqn.load_state_dict(self.main_dqn.state_dict())
                    steps_to_update_target_model = 0

                if steps_to_decrease_e_greedy == self.steps_to_decrease_e_greedy:
                    if self.e_greedy > self.e_min:
                        self.e_greedy *= self.e_decay
                    steps_to_decrease_e_greedy = 0

                if done:

                    print(f"Episode {episode + 1 - self.batch_size}/{episodes} [{steps} steps]: eps_greedy={self.e_greedy:.3f} "
                          f"Accumulated reward={accum_reward:.3f} Loss={np.mean(losses):.5f}")
                    break

                state = next_state
                steps += 1

        torch.save(self.main_dqn.state_dict(), 'optimal.pth')

    def replay(self, batch) -> torch.Tensor:
        running_loss = torch.tensor(0, dtype=torch.float32)
        for e in batch:
            y = e.reward
            if not e.done:
                y = (e.reward + self.gamma * torch.max(self.target_dqn(e.next_state)))
            e_loss = ((y - self.main_dqn(e.state)[0, e.action]) ** 2) / len(batch)
            running_loss += e_loss
        return running_loss

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    # reproducibility
    RANDOM_SEED = 5
    torch.random.manual_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # continuous observations and discrete actions
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    main_dqn = DQN(obs_dim, act_dim)
    target_dqn = DQN(obs_dim, act_dim)
    dqn_trainer = DQNTrainer(main_dqn, target_dqn, env)
    dqn_trainer.train(1000)







