from pathlib import Path

import gym
import numpy as np
import torch
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent
from smarts.env.custom_observations import lane_ttc_observation_adapter
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from ray.rllib.agents import ppo
from custom_env import Wrapper

# This action space should match the input to the action_adapter(..) function below.
ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
)


# This observation space should match the output of observation_adapter(..) below
OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)


def observation_adapter(env_observation):
    return lane_ttc_observation_adapter.transform(env_observation)


def reward_adapter(env_obs, env_reward):
    return env_reward


def action_adapter(model_action):
    throttle, brake, steering = model_action
    return np.array([throttle, brake, steering * np.pi * 0.25])


class TrainingModel(FullyConnectedNetwork):
    NAME = "FullyConnectedNetwork"


ModelCatalog.register_custom_model(TrainingModel.NAME, TrainingModel)


class RLLibTorchModelAgent(Agent):
    def __init__(self, observation_space, action_space, model_config):
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self.model = TrainingModel(self._prep.observation_space, action_space, num_outputs=3, model_config=model_config, name="Name")

    def act(self, obs):
        obs = self._prep.transform(obs)
        obs = obs[np.newaxis, :]
        obs = torch.tensor(obs)
        action = self.model({"obs": obs})[0]
        return list(action.flatten().detach().cpu().numpy())


rllib_agent = {
    "agent_spec": AgentSpec(
        interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=500),
        agent_params={
            "observation_space": OBSERVATION_SPACE,
            "action_space": ACTION_SPACE,
            "model_config": {
                "fcnet_activation": "relu",
                "fcnet_hiddens": [64, 64]
            }
        },
        agent_builder=RLLibTorchModelAgent,
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
        action_adapter=action_adapter,
    ),
    "observation_space": OBSERVATION_SPACE,
    "action_space": ACTION_SPACE,
}


if __name__ == "__main__":

    rllib_policies = {
        "default_policy": (
            None,
            rllib_agent["observation_space"],
            rllib_agent["action_space"],
            {"model": {"custom_model": TrainingModel.NAME}},
        )
    }

    tune_config = {
        "env": Wrapper,
        "framework": "torch",
        "log_level": "INFO",
        "num_workers": 2,
        "env_config": {
            "base_env_cls": RLlibHiWayEnv,
            "seed": 42,
            "scenarios": ["/home/rudy/PycharmProjects/playground/scenarios/intersections/4lane"],
            "headless": True,
            "agent_specs": {
                "AGENT-1": rllib_agent["agent_spec"]
            },
            "action_adapter": action_adapter
        },
        "multiagent": {"policies": rllib_policies},
    }

    # ray.init()
    # config = ppo.DEFAULT_CONFIG.copy()
    # config["env_config"] = tune_config["env_config"]
    # config["multiagent"] = tune_config["multiagent"]
    # config["framework"] = "torch"
    # agent = ppo.PPOTrainer(config, env=tune_config["env"])
    # agent.train()

    ray.init()
    analysis = tune.run(
        "PPO",
        name="PPOExp",
        stop={"time_total_s": 100},
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir="/home/rudy/PycharmProjects/playground/results",
        max_failures=3,
        config=tune_config,
        verbose=1
    )



