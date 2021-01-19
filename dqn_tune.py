from ray import tune
import ray
from dqn import *


def get_trainable(main_dqn, target_dqn, env, n_iterations):
    def trainable(config):
        t = DQNTrainer(main_dqn, target_dqn, env, config)
        for step in range(n_iterations):
            min_reward, mean_reward, max_reward, _ = t.train_iter()
            tune.report(mean_reward=mean_reward, max_reward=max_reward, min_reward=min_reward)

    return trainable


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
    main_dqn.train()
    target_dqn.train()
    dqn_trainable = get_trainable(main_dqn, target_dqn, env, n_iterations=100)

    config = DEFAULT_CONFIG.copy()
    config['local_dir'] = '/home/rudy/PycharmProjects/playground/results/CartPole'
    config['verbose'] = False
    config['steps_per_iter'] = 1000  # this shouldn't affect learning
    config['steps_to_update_main_model'] = 1
    config['steps_to_update_target_model'] = 250
    config['learning_rate'] = 0.001
    config['gamma'] = 0.9
    config['e_decay'] = 0.995
    print(config)

    reporter = tune.CLIReporter()
    reporter.add_metric_column('min_reward')
    reporter.add_metric_column('mean_reward')
    reporter.add_metric_column('max_reward')

    ray.init()
    analysis = tune.run(
        dqn_trainable,
        name="CartPole",
        config=config,
        verbose=1,
        local_dir='/home/rudy/PycharmProjects/playground/results/CartPoleMyDQN'
        # resources_per_trial={'gpu': 1}
    )
    print(analysis)
