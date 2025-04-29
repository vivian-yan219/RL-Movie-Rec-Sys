import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory
from tf_agents.bandits.agents import linear_thompson_sampling_agent
from tf_agents.bandits.agents.neural_epsilon_greedy_agent import NeuralEpsilonGreedyAgent
from tf_agents.specs import tensor_spec
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.networks import network
from tensorflow.keras import layers

from movielens_bandit_env import RealMovieLensEmbeddingEnv

# Reward Network
class SimpleRewardNetwork(network.Network):
    def __init__(self, input_tensor_spec, output_tensor_spec, name='RewardNetwork'):
        super().__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)
        self._num_actions = output_tensor_spec.maximum - output_tensor_spec.minimum + 1
        self._model = tf.keras.Sequential([
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            layers.Dense(self._num_actions)
        ])

    def call(self, observation, step_type=None, network_state=(), training=False):
        output = self._model(observation, training=training)
        return output, network_state

# Create agent
def create_agent(agent_type, env):
    if agent_type == "LinearThompson":
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            epsilon=0.3,  # Fixed epsilon exploration
            emit_policy_info=(),
            accepts_per_arm_features=False
        )
    elif agent_type == "Random":
        return RandomTFPolicy(env.time_step_spec(), env.action_spec())
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

# Top-K Evaluation
def evaluate_agent_topk(agent, raw_env, k=5, num_users=50):
    if not hasattr(agent, '_reward_network') or agent._reward_network is None:
        print("Skip evaluation: Agent does not use a reward network.")
        return 0, 0, 0, 0

    precisions, recalls, ndcgs, maps = [], [], [], []

    num_users = min(num_users, raw_env._user_embeddings.shape[0])
    user_ids = np.random.choice(num_users, size=num_users, replace=False)

    for user_id in user_ids:
        user_embedding = raw_env._user_embeddings[user_id]
        user_embedding = tf.convert_to_tensor(user_embedding.reshape(1, -1), dtype=tf.float32)

        try:
            scores, _ = agent._reward_network(user_embedding)
            scores = scores.numpy().flatten()
        except Exception:
            continue

        top_k_indices = np.argsort(scores)[::-1][:k]
        relevant_items = raw_env.get_user_liked_items(user_id)
        if len(relevant_items) == 0:
            continue

        relevant = set(relevant_items)
        recommended = top_k_indices.tolist()

        precision = len(set(recommended) & relevant) / k
        recall = len(set(recommended) & relevant) / len(relevant)

        def dcg(items, relevant):
            return sum([1.0 / np.log2(i + 2) if item in relevant else 0.0 for i, item in enumerate(items)])

        ideal_dcg = dcg(sorted(relevant), relevant)
        ndcg = dcg(recommended, relevant) / ideal_dcg if ideal_dcg > 0 else 0.0

        def average_precision(items, relevant):
            hits, sum_precisions = 0, 0
            for i, item in enumerate(items):
                if item in relevant:
                    hits += 1
                    sum_precisions += hits / (i + 1)
            return sum_precisions / min(len(relevant), k)

        ap = average_precision(recommended, relevant)

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        maps.append(ap)

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs), np.mean(maps)

# Main training loop
def run_experiment(agent_type="EpsilonGreedy", steps=50000):
    print(f"Running experiment with agent: {agent_type}")

    train_env = tf_py_environment.TFPyEnvironment(
        RealMovieLensEmbeddingEnv(num_users=943, num_movies=1682, embedding_dim=16, split='train')
    )
    test_env = tf_py_environment.TFPyEnvironment(
        RealMovieLensEmbeddingEnv(num_users=943, num_movies=1682, embedding_dim=16, split='test')
    )
    raw_test_env = test_env.pyenv._envs[0]

    agent = create_agent(agent_type, train_env)
    agent.initialize()

    reward_history = []
    precision_history = []
    recall_history = []
    ndcg_history = []
    map_history = []

    time_step = train_env.reset()

    for step in range(steps):
        # ==== Add epsilon decay here ====
        if agent_type == "EpsilonGreedy":
            if step < 10000:
                agent.policy._epsilon = 0.3
            elif step < 30000:
                agent.policy._epsilon = 0.1
            elif step < 50000:
                agent.policy._epsilon = 0.05
            else:
                agent.policy._epsilon = 0.01

        action_step = agent.policy.action(time_step)

        next_time_step = train_env.step(action_step.action)

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
        time_step = train_env.reset()

        if (step + 1) % 100 == 0:
            mean_reward = np.mean(reward_history)
            print(f"Step {step+1:5d} | current reward: {reward:.2f} | accumulated reward: {mean_reward:.4f}")

        if (step + 1) % 1000 == 0:
            precision, recall, ndcg, map_score = evaluate_agent_topk(agent, raw_test_env, k=5, num_users=50)
            precision_history.append(precision)
            recall_history.append(recall)
            ndcg_history.append(ndcg)
            map_history.append(map_score)
            print(f"Step {step+1} | Precision@5: {precision:.4f} | Recall@5: {recall:.4f} | NDCG@5: {ndcg:.4f} | MAP@5: {map_score:.4f}")

    # Plot Reward
    window = 100
    rolling = np.convolve(reward_history, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, alpha=0.3, label="Instant Reward")
    plt.plot(range(window-1, len(reward_history)), rolling, label=f"{window}-step Rolling Avg", linewidth=2)
    plt.title(f"{agent_type} on Real MovieLens - Reward Curve")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot TopK metrics
    x = np.arange(1000, 1000*len(precision_history)+1, 1000)
    plt.figure(figsize=(10,5))
    plt.plot(x, precision_history, label="Precision@5")
    plt.plot(x, recall_history, label="Recall@5")
    plt.plot(x, ndcg_history, label="NDCG@5")
    plt.plot(x, map_history, label="MAP@5")
    plt.title("Top-K Metrics over Training")
    plt.xlabel("Training Steps")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return agent


if __name__ == "__main__":
    agent = run_experiment(agent_type="EpsilonGreedy", steps=100000)

    # Final full evaluation
    print("\nFinal evaluation on full test set...")
    test_env = tf_py_environment.TFPyEnvironment(
        RealMovieLensEmbeddingEnv(num_users=943, num_movies=1682, embedding_dim=16, split='test')
    )
    raw_test_env = test_env.pyenv._envs[0]
    precision, recall, ndcg, map_score = evaluate_agent_topk(agent, raw_test_env, k=5, num_users=50)
    print(f"\nFinal Top-5 Results:")
    print(f"Precision@5: {precision:.4f}")
    print(f"Recall@5:    {recall:.4f}")
    print(f"NDCG@5:      {ndcg:.4f}")
    print(f"MAP@5:       {map_score:.4f}")
