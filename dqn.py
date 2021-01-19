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


DEFAULT_CONFIG = {
        'e_greedy': 1,
        'e_min': 0.01,
        'e_decay': 0.995,
        'gamma': 0.9,
        'steps_per_iter': 100,
        'steps_to_update_main_model': 1,
        'steps_to_update_target_model': 500,
        'steps_to_decrease_e_greedy': 50,
        'checkpoint_frequency': 5,
        'replay_buffer_size': 1000,
        'local_dir': 'results/PPO',
        'batch_size': 32,
        'learning_rate': 0.0001,
        'verbose': True
    }


class DQNTrainer:
    def __init__(self, main_dqn: DQN, target_dqn: DQN, env: gym.Env, config: dict):
        self.env = env
        # rl
        self.main_dqn = main_dqn
        self.target_dqn = target_dqn
        self.e_greedy = config['e_greedy']  # greedy strategy
        self.e_min = config['e_min']  # min epsilon value
        self.e_decay = config['e_decay']  # epsilon decay
        self.gamma = config['gamma']  # gamma factor for discounted rewards

        # training
        self.steps_per_iter = config['steps_per_iter']  # number of steps per iteration
        self.steps_to_update_main_model = config[
            'steps_to_update_main_model']  # number of steps to update main (online or behaviour) model
        self.update_steps = 0
        self.steps_to_update_target_model = config[
            'steps_to_update_target_model']  # number of steps to update target model
        self.greedy_steps = 0
        self.steps_to_decrease_e_greedy = config[
            'steps_to_decrease_e_greedy']  # number of steps to decrease the epsilon value
        self.checkpoint_freq = config['checkpoint_frequency']  # number of iterations to save the target model
        self.buffer = ExperienceReplay(config['replay_buffer_size'])  # buffer size
        self.checkpoint_path = config['local_dir']  # path where checkpoints are going to be stored

        # learning
        self.loss = torch.nn.MSELoss()
        self.batch_size = config['batch_size']  # size of each mini-batch used for training
        self.optimizer = torch.optim.Adam(self.main_dqn.parameters(), lr=config['learning_rate'])  # optimizer
        self.verbose = config['verbose']

    def train(self, iterations: int):
        """
        Train a DQN for N iterations. One iteration corresponds to some defined number of steps.
        :param iterations:
        :return:
        """

        s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} epsilon {:1.3f} {}"
        s_check = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} epsilon {:1.3f} saved {} "
        total_steps = 0
        iter_metrics = []
        for n in range(iterations):
            r_min, r_mean, r_max, iter_steps = self.train_iter()
            iter_metrics.append((r_min, r_mean, r_max))
            total_steps += iter_steps

            if n == int(iterations / 2):
                self.steps_to_update_target_model = int(self.steps_to_update_target_model / 2)

            # checkpointing & logging
            s_print = s
            file_name = ""
            if n % self.checkpoint_freq == 0:
                file_name = f'my_dqn_{n}.pth'
                torch.save(self.target_dqn.state_dict(), os.path.join(self.checkpoint_path, file_name))
                s_print = s_check

            if self.verbose:
                print(s_print.format(
                    n + 1,
                    r_min,
                    r_mean,
                    r_max,
                    total_steps,
                    self.e_greedy,
                    file_name
                ))
        iter_min = np.mean([x[0] for x in iter_metrics])
        iter_mean = np.mean([x[1] for x in iter_metrics])
        iter_max = np.mean([x[2] for x in iter_metrics])
        return iter_min, iter_mean, iter_max

    def train_iter(self):
        episode = 0  # Episode counter
        iter_steps = 0  # Global steps in actual iteration
        episode_rewards = []  # stores the rewards obtained at each episode

        while iter_steps < self.steps_per_iter:
            state = self.env.reset()
            losses = []
            rewards = []
            # region: RUN EPISODE
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
                    # optimization step
                    batch_trans = self.buffer.sample(self.batch_size)
                    loss = self.replay(batch_trans)
                    losses.append(loss.item())
                    self.update(loss)

                if self.update_steps >= self.steps_to_update_target_model:
                    if self.verbose:
                        print('Copying main network weights to the target network weights')
                    self.target_dqn.load_state_dict(self.main_dqn.state_dict())
                    self.update_steps = 0

                if self.greedy_steps >= self.steps_to_decrease_e_greedy:
                    if self.e_greedy > self.e_min:
                        self.e_greedy *= self.e_decay
                    self.greedy_steps = 0

                if done:
                    episode_rewards.append(rewards)
                    episode += 1
                    # print(
                    #     f"Episode {episode} [{len(rewards)} steps]: eps_greedy={self.e_greedy:.3f} "
                    #     f"Accumulated reward={np.sum(rewards):.3f} Loss={np.mean(losses):.5f}")
                    break

                state = next_state
                self.greedy_steps += 1
                self.update_steps += 1
                iter_steps += 1
            # endregion

        episode_rewards = [np.sum(r) for r in episode_rewards]
        episodes_min_reward = np.min(episode_rewards)
        episode_max_reward = np.max(episode_rewards)
        episode_mean_reward = np.mean(episode_rewards)
        return episodes_min_reward, episode_mean_reward, episode_max_reward, np.sum(episode_rewards)

    def replay(self, batch) -> torch.Tensor:
        """
        Self implementation of mean squared loss
        """
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
        y_real = torch.tensor(y_real, requires_grad=False)
        y_real.detach_() # avoid propagating gradients to the target network
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
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    main_dqn = DQN(obs_dim, act_dim)
    target_dqn = DQN(obs_dim, act_dim)
    # freeze target network to ensure proper learning
    for parameter in target_dqn.parameters():
        parameter.requires_grad = False
    main_dqn.train()
    target_dqn.train()
    dqn_trainer = DQNTrainer(main_dqn, target_dqn, env, DEFAULT_CONFIG)
    dqn_trainer.train(1000)
