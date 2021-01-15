import torch
import gym
import random
import numpy as np
import os


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
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.fc2 = torch.nn.Linear(256, output_dim)

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
        # rl
        self.main_dqn = main_dqn
        self.target_dqn = target_dqn
        self.e_greedy = 1.0
        self.e_min = 0.01
        self.e_decay = 0.9
        self.gamma = 0.999

        # training
        self.steps_per_iter = 200
        self.steps_to_update_main_model = 2
        self.steps_to_update_target_model = 500
        self.steps_to_decrease_e_greedy = 200
        self.checkpoint_freq = 10
        self.buffer = ExperienceReplay(50000)
        self.checkpoint_path = 'results/MyDQN'

        # learning
        self.loss = torch.nn.MSELoss()
        self.batch_size = 32
        self.optimizer = torch.optim.Adam(self.main_dqn.parameters(), lr=0.0005)

    def train(self, iterations: int):

        s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} epsilon {:1.3f} {}"
        s_check = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} epsilon {:1.3f} saved {} "
        total_steps = 0
        for n in range(iterations):
            r_min, r_mean, r_max, iter_steps = self.train_iter()
            total_steps += iter_steps
            s_print = s
            file_name = ""
            if n % self.checkpoint_freq == 0:
                file_name = f'my_dqn_{n}.pth'
                torch.save(self.main_dqn.state_dict(), os.path.join(self.checkpoint_path, file_name))
                s_print = s_check

            print(s_print.format(
                n + 1,
                r_min,
                r_mean,
                r_max,
                total_steps,
                self.e_greedy,
                file_name
            ))

    def train_iter(self):
        episode = 0  # Episode counter
        iter_steps = 0  # Global steps in actual iteration
        steps_to_update_target_model = 0  # Counter to update target model
        steps_to_decrease_e_greedy = 0  # Counter to decrease epsilon greedy value
        episode_rewards = []
        while iter_steps < self.steps_per_iter:
            state = self.env.reset()
            losses = []
            rewards = []
            while True:
                # epsilon greedy strategy
                if random.random() < self.e_greedy:
                    action = self.env.action_space.sample()
                else:
                    action = torch.argmax(self.main_dqn(state)).item()

                # perform action
                next_state, r, done, _ = self.env.step(action)
                rewards.append(r)
                new_trans = Experience(state, next_state, action, r, done)
                self.buffer.put(new_trans)

                if len(self.buffer) < self.batch_size:
                    break

                if iter_steps % self.steps_to_update_main_model == 0:
                    batch_trans = self.buffer.sample(self.batch_size)
                    loss = self.replay(batch_trans)
                    losses.append(loss.item())
                    self.update(loss)

                if steps_to_update_target_model >= self.steps_to_update_target_model:
                    # print('Copying main network weights to the target network weights')
                    self.target_dqn.load_state_dict(self.main_dqn.state_dict())
                    steps_to_update_target_model = 0

                if steps_to_decrease_e_greedy == self.steps_to_decrease_e_greedy:
                    if self.e_greedy > self.e_min:
                        self.e_greedy *= self.e_decay
                    steps_to_decrease_e_greedy = 0

                if done:
                    episode_rewards.append(rewards)
                    episode += 1
                    # print(
                    #     f"Episode {episode} [{len(rewards)} steps]: eps_greedy={self.e_greedy:.3f} "
                    #     f"Accumulated reward={np.sum(rewards):.3f} Loss={np.mean(losses):.5f}")
                    break

                state = next_state
                steps_to_update_target_model += 1
                steps_to_decrease_e_greedy += 1
                iter_steps += 1

        episode_rewards = [np.sum(r) for r in episode_rewards]
        episodes_min_reward = np.min(episode_rewards)
        episode_max_reward = np.max(episode_rewards)
        episode_mean_reward = np.mean(episode_rewards)
        return episodes_min_reward, episode_mean_reward, episode_max_reward, np.sum(episode_rewards)

    def replay(self, batch) -> torch.Tensor:
        running_loss = torch.tensor(0, dtype=torch.float32)
        for e in batch:
            y = e.reward
            if not e.done:
                y = (e.reward + self.gamma * torch.max(self.target_dqn(e.next_state)))
            e_loss = ((y - self.main_dqn(e.state)[0, e.action]) ** 2) / len(batch)
            running_loss += e_loss
        return running_loss

    def replay2(self, batch):
        """
        Torch implementation of mean squared loss
        """
        y_real, y_predicted = [], []
        for e in batch:
            y_real_j = e.reward
            if not e.done:
                y_real_j = (e.reward + self.gamma * torch.max(self.target_dqn(e.next_state)))
            y_predicted_j = self.main_dqn(e.state)[0, e.action]
            y_real.append(y_real_j)
            y_predicted.append(y_predicted_j)
        y_real = torch.tensor(y_real, requires_grad=True)
        y_predicted = torch.tensor(y_predicted, requires_grad=True)
        return self.loss(y_predicted, y_real)

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
    main_dqn.train()
    target_dqn.train()
    dqn_trainer = DQNTrainer(main_dqn, target_dqn, env)
    dqn_trainer.train(1000)
