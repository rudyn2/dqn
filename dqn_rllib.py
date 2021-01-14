from ray import tune
import yaml
import ray


with open('cart-pole-dqn-rllib.yaml', 'r') as f:
    config = yaml.load(f)

config['config']['n_step'] = 10

ray.init()
tune.run(
    "DQN",
    name="DQN-420-stop",
    stop=config['stop'],
    local_dir='.',
    config=config['config'],
    checkpoint_at_end=True
)
