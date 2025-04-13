import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.policies.random_tf_policy import RandomTFPolicy

from tf_agents.bandits.agents import lin_ucb_agent, linear_thompson_sampling_agent
from tf_agents.bandits.agents.greedy_reward_prediction_agent import GreedyRewardPredictionAgent
from tf_agents.bandits.agents.neural_epsilon_greedy_agent import NeuralEpsilonGreedyAgent


from movielens_bandit_env import RealMovieLensEmbeddingEnv


from tf_agents.networks import network
from tensorflow.keras import layers

class SimpleRewardNetwork(network.Network):
    def __init__(self, input_tensor_spec, output_tensor_spec, name='RewardNetwork'):
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name
        )
        self._model = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_tensor_spec.maximum - output_tensor_spec.minimum + 1)
        ])

    def call(self, observation, step_type=None, network_state=(), training=False):
        output = self._model(observation)
        return output, network_state


class RandomAgentWrapper:
    def __init__(self, time_step_spec, action_spec):
        self.policy = RandomTFPolicy(time_step_spec, action_spec)

    def initialize(self):
        pass

    def train(self, *args, **kwargs):
        pass



def create_agent(agent_type, env):
    if agent_type == "LinearUCB":
        return lin_ucb_agent.LinearUCBAgent(
            time_step_spec=env.time_step_spec(),
            action_spec=env.action_spec(),
            variable_collection=None,
            accepts_per_arm_features=False,
            alpha=tf.Variable(1.0, trainable=False, dtype=tf.float32)
        )

    elif agent_type == "LinearThompson":
        return linear_thompson_sampling_agent.LinearThompsonSamplingAgent(
            time_step_spec=env.time_step_spec(),
            action_spec=env.action_spec(),
            dtype=tf.float32
        )

    elif agent_type == "EpsilonGreedy":

        reward_net = SimpleRewardNetwork(
            input_tensor_spec=env.observation_spec(),
            output_tensor_spec=env.action_spec()
        )

        return NeuralEpsilonGreedyAgent(
            time_step_spec=env.time_step_spec(),
            action_spec=env.action_spec(),
            reward_network=reward_net,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            epsilon=0.1,
            emit_policy_info=(),
            accepts_per_arm_features=False
        )

    elif agent_type == "Greedy":
        reward_net = SimpleRewardNetwork(
            input_tensor_spec=env.observation_spec(),
            output_tensor_spec=env.action_spec()
        )

        return GreedyRewardPredictionAgent(
            time_step_spec=env.time_step_spec(),
            action_spec=env.action_spec(),
            reward_network=reward_net,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            accepts_per_arm_features=False
        )

    elif agent_type == "Random":
        return RandomAgentWrapper(
            time_step_spec=env.time_step_spec(),
            action_spec=env.action_spec()
        )

    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")



def run_experiment(agent_type="LinearUCB", steps=300):
    print(f"Running experiment with agent: {agent_type}")
    env = tf_py_environment.TFPyEnvironment(
        RealMovieLensEmbeddingEnv(num_users=50, num_movies=50, embedding_dim=16)
    )

    agent = create_agent(agent_type, env)
    agent.initialize()

    reward_history = []
    time_step = env.reset()

    for step in range(steps):
        raw_env = env.pyenv._envs[0]
        current_rating = raw_env._ratings[raw_env._index]

        action_step = agent.policy.action(time_step)
        next_time_step = env.step(action_step.action)


        if hasattr(agent, "train"):
            exp = trajectory.Trajectory(
                step_type=tf.expand_dims(time_step.step_type, axis=1),
                observation=tf.expand_dims(time_step.observation, axis=1),
                action=tf.expand_dims(action_step.action, axis=1),
                policy_info=action_step.info,
                next_step_type=tf.expand_dims(next_time_step.step_type, axis=1),
                reward=tf.expand_dims(next_time_step.reward, axis=1),
                discount=tf.expand_dims(next_time_step.discount, axis=1)
            )
            agent.train(exp)

        reward = next_time_step.reward.numpy()[0]
        reward_history.append(reward)
        time_step = env.reset()

        if (step + 1) % 10 == 0:
            print(f"Step {step + 1:3d} | action: {action_step.action.numpy()[0]} | rating: {current_rating:.1f} | reward: {reward:.2f}")

        if (step + 1) % 25 == 0:
            print(f"Step {step+1:3d} | current reward: {reward:.2f} | accumulated reward: {np.mean(reward_history):.4f}")


    window = 20
    rolling = np.convolve(reward_history, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, alpha=0.3, label="Instant Reward")
    plt.plot(range(window - 1, len(reward_history)), rolling, label=f"{window}-step Rolling Avg", linewidth=2)
    plt.title(f"{agent_type} on Real MovieLens")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


run_experiment(agent_type="LinearUCB")
#run_experiment(agent_type="LinearThompson")
#run_experiment(agent_type="EpsilonGreedy")
#run_experiment(agent_type="Greedy")
# run_experiment(agent_type="Random")
