# MovieLens Recommendation With Actor–Critic (DDPG)

This notebook implements a Deep Deterministic Policy Gradient (DDPG) recommender on the MovieLens dataset.

## Contents

1. **Import Libraries**  
   Load pandas, numpy, TensorFlow 1.x, etc.

2. **Data Preprocessing**  
   - Read & merge ratings and movie titles  
   - Build per-user chronological rating histories

3. **Movie Embeddings**  
   - Generate and save dense vectors for each movie  
   - Helpers for loading train/test splits and embeddings

4. **Environment Simulator**  
   - Model recommendations as an MDP (state = user history, action = movie recommendation)

5. **Actor Network**  
   - GRU-based policy mapping state sequences → action embeddings  
   - Online/target networks with soft updates

6. **Critic Network**  
   - GRU + fully connected Q-function approximator for (state, action) pairs  
   - Online/target critics with MSE loss

7. **Replay Buffer**  
   - Store transitions (state, action, reward, next state) for experience replay

8. **Training Loop**  
   - Jointly train Actor & Critic over multiple epochs  
   - Use target networks and policy gradients

9. **Testing**  
   - Evaluate on held-out users

10. **Performance Evaluation**  
    - Ranking metrics on train/test sets (e.g., precision@K, recall@K)

## Usage

1. Install required packages.  
2. Run cells sequentially to preprocess data, train models, and evaluate.  
3. Tweak hyperparameters in the Actor/Critic sections as needed.  
