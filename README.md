# RL-Movie-Rec-Sys
DS-GA 3001 Reinforcement Learning Project Spring 2025

## Introduction
The objective of this project is to explore the application of RL in personalized recommendation systems. In this GitHub, we developed a custom RL environment based on the Movielens-100k dataset that simulates realistic user interactions. Building on this environment, we implemented and experiment with four different reinforcement learning algorithms, including bandits, DQN, and actor-critic methods. The performance of these models is then systematically evaluated using offline metrics, such as precision, recall, ndcg, and MAP.

## Methodology

### Bandit

-----

### DQN

**Type**: off-policy, value function approximator

**Key ideas**: We built a DQN agent from `keras-rl`, incorporating buffer replay memory and epsilon greedy policy (with epsilon decay). We trained the model with 10,000 steps for the 100k data.
  - Load data and data preprocessing: return three datasets (ratings, movies, users) for future usage.
  - Create a RL environment, where:
    - States: a history of watched movies, where 10 past movies are remembered.
    - Actions: discrete - recommend a new available movie by selecting its index.
    - Rewards: +1 if the user liked the recommended movie with rating>=4, otherwise 0.
  - Build a DQN agent based on `keras-rl`, where:
    - DQN agent: use movie embeddings as input, add two dense layers, use linear activation for final layer.
    - Configuration: apply a standard replay buffer with 50,000 experiences, epsilon greedy exploration with linear decay.
    - Compilation: use double DQN to avoid overestimating Q-values with Adam and mean absolute error.
  - Training, evaluations using top k ranking metrics, and improvements by modifying model structure, exploration policy, and optimizer.
  - 
-----

### DDPG

**Type**: off-policy, model-free, actor-critic

**Key ideas**: We performed two different experiments regarding the DDPG agent, which can be found under the folders DDPG and Actor_Critic respectively.
- **DDPG**: We built a DDPG agent from scratch, using the actor and critic network with pretrained embeddings, reducing overestimated Q-values, and applying priorized experience replay (PER). The saved model of actor and critic are generated after training is done: `python train.py`. During evaluation, we experimented with the saved models and recommended related movies with respect to user's watch history.
  - Key RL components initialization:
    - `env.py`: an offline interface to interact with users and movie recommendations.
    - `actor.py`: takes the state and outputs a continuous action, which represents user's movie preference vector.
    - `critic.py`: takes (state, action) and predicts Q-value, which is the expected future reward.
    - `state_representation.py`: averages the embeddings of past interacted movies and the user embedding to form the state vector.
    - `replay_buffer.py`: stores past experiences `replay_memory.py` with priority sampling (important experiences are replayed more often) using a tree structure `tree.py`.
  - Create MovieGenre (not used) and UserMovie embedding model `embedding.py` that embeds user_id and movie_id into dense vectors.
    - `embed_model.ipynb`: Train the embedding model with Adam and binary cross entropy (BCE), where Final loss is 0.5675 and accuracy is 71.5.
  - Build a DDPG agent, where:
    - Initialize actor and critic models, user/movie embedding network with pre-trained weights, state representation model, and priority replay buffer (PER). Set epsilon decay schedule for exploration.
    - Given the current action vector, compute dot product between action and all available movie embeddings and recommend the highest scoring movie(s) that haven't been seen yet.
  - Train `train_ddpg.py` and evaluate `eval_ddpg.ipynb`.

    **Hyperparameters** (example)  
    | Parameter       | Value     |
    |-----------------|-----------|
    | Actor hidden size  | 128     |
    | Critic hidden size | 128     |
    | Embedding dimension     | 100      |
    | Replay buffer size    | 100,000    |
    | Discount factor   | 256       |
    | Batch size      | 32    |
    | Target network soft update | 0.001  |
    | Initial exploration noise std dev| 1.5|

- **Approach 2: Actor_Critic**: The second approach can be found in the folder Actor_Critic. The Jupyter Notebook implements an actor-critic deep reinforcement learning approach (DDPG) for personalized movie recommendations using the MovieLens 100K dataset. 
  - In this notebook, we model the recommendation problem as a Markov Decision Process (MDP) and apply the Deep Deterministic Policy Gradient (DDPG) algorithm. The workflow includes:
    - Generating embeddings for movie items via a neural network classifier.
    - Defining an environment simulator for user–movie interactions.
    - Implementing an actor network that proposes recommendation actions based on user history.
    - Implementing a critic network that evaluates the quality of actions (Q-values).
    - Training with experience replay to stabilize learning.
    - Evaluating recommendation performance on a held-out test set.
  - Hyperparameters
    - history_length: length of user history sequence (20)
    - ra_length: number of items recommended per step (10)
    - buffer_size: size of replay memory (1000000)
    - batch_size: mini-batch size for learning (128)
    - discount_factor: reward discount factor γ (0.99)
    - nb_episodes: number of training episodes (5)

-----

### Proximal Policy Optimization (PPO)

**Type**: on-policy, policy-gradient, actor-critic

**Key ideas**:  
- Use a clipped surrogate objective to constrain policy updates  
- Maintain a value function baseline for variance reduction  
- Alternate between collecting rollouts and performing multiple epochs of minibatch SGD
 
**Hyperparameters** (example)  
| Parameter       | Value     |
|-----------------|-----------|
| Learning rate   | 1e-4      |
| Discount factor | 0.98      |
| Clip range      | 0.1       |
| Entropy :    | 0.01      |
| n_steps         | 256       |
| Batch size      | 128       |
| Total timesteps | 200 000   |

**Environment ('MovieRecRnv')**
Each episode is a sequence of movie recommendations up to 'max_steps':

- **Observation vector** (dimension ≈ `1 + 1 + num_genres + 7 + num_occupations + 2`):  
  1. Normalized user mean rating  
  2. Normalized movie mean rating  
  3. One-hot genre vector  
  4. Age bucket (7-dim one-hot)  
  5. Occupation one-hot  
  6. Gender one-hot (M/F)
 
- **Action space**: Discrete set of all `movie_id`s in the catalog  
- **Reward**:  
  - +1.0 for exact rating match  
  - –|predicted_rating – true_rating| otherwise
 
- **Episode termination**: after `max_steps` recommendations or exhaust the dataset

**File Structure & Setup Guide**

- `utils/data_loader.py`:
  - Reads and cleans the MovieLens dataset, renames columns, one-hot encodes genres, and returns DataFrames ready for the environment
- `env/movie_rec_env.py` defines MovieRecEnv, which is a Gymnasium environment that:
  - Builds per-step feature vectors from user/movie metadata
  - Exposes a discrete action space (predict a rating 1 - 5)
  - Implements a reward function
  - Terminates episodes after a fixed number of steps or dataset exhaustion
- `train/ppo_train.py`:
  - Loads data via `data_loader.py`, creates a vectorized and normalized environment (`DummyVecEnv` + `VecNOrmalize`), instantiates a PPO agent with tuned hyperparameters, runs training for a specified number of timesteps, and saves both the model weights and normalization statistics under `models/`
- `train/evaluate.py` loads the saved PPO policy and VecNormalize stats, reconstructs the environment, then:
  - Runs episodes to report the average cumulative reward
  - Splits out a test portion of the ratings
  - Scores every candidate movie per user to compute ranking metrics: Precision@K, Recall@K, NDCG@K, and Mean Average Precision@K
- `main.py` ties everything together depending on ``--mode``:
  - `--mode=train` invoke `train/ppo_train.py` with chosen `--timesteps` and `--max_steps`
  - `--mode=eval` invoke `train/evaluate.py` with chosen `--episodes` and `--max_steps`, and ranking cutoff `K`, then print out all metrics

## Results
||Precision@10|Recall@10|NDCG@10|MAP@10|
|-|-|-|-|-|
|Contextual MAB|0.0163|0.0315|0.0237|0.0174|
|DQN|0.0300|0.0054|0.0660|0.3000|
|DDPG|0.0102|0.0016|0.0132|0.0451|
|Actor Critic|0.0693|0.0182|0.0759|0.0094
|PPO|0.1839|0.1332|0.1893|0.0465|
|..|||||
