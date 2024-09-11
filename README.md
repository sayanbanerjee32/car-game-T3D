# Car driving simulation using TD3 

This repo demonstrates how to train a reinforcement learning model using Twin-Delayed Deep Deterministic (TD3) policy gradient. A city map and kivy package is used to create the environment for simulation. 

## New environment 
![alt text](images_new/citymap.png)   

The above city map is used for training the simulation. Following mask is created to map the roads.   

![alt text](images_new/MASK1.png)   

There are 3 destinations on the map which are randomly selected in each turn and once selected shown with following icon on the map.   

![alt text](images_new/target.png)   

## TD3 Model Training Techniques

A key feature of this model is its continuous action space. Instead of selecting from predefined options, the model predicts a continuous value used directly as an action. Specifically, it predicts an angle value to rotate the car, rather than choosing from a set of discrete angles. This continuous action space makes the training process more challenging. To address this, we've implemented several techniques and configurations:

1. **Replay Buffer**: We use a large replay buffer (500,000 samples) to store experiences. This ensures a diverse set of samples for training, including both positive and negative rewards.

2. **Initial Exploration**: The model starts training only after the replay buffer has collected a sufficient number of samples (defined by `initial_buffer`). This allows for a good mix of experiences before learning begins.

3. **Stuck Detection and Reset**: If the car remains in the same position for too long (defined by `stuck_patience`), it's considered stuck and is reset to a new random position. This prevents the agent from getting trapped in suboptimal states.

4. **Balanced Sampling**: Our `ReplayBuffer` class is designed to maintain a balance between positive and negative examples. It uses a `positive_sample_ratio` to ensure that each training batch contains a good mix of rewarding and challenging experiences.

5. **Adaptive Exploration**: The TD3 algorithm uses noise for exploration, which is gradually reduced over time. This allows for more random actions initially and more exploitative actions as the agent learns.

6. **Periodic Model Saving**: The model is saved at regular intervals (defined by `save_interval`). This allows for recovery of the best-performing model if training degrades over time.

7. **Flexible Running Modes**: The system supports different running modes (train, load, inference) controlled by the `RUN_MODE` environment variable. This allows for easy switching between training and evaluation.

8. **Dynamic Goal Setting**: When the car reaches a goal, a new random goal is set. This continually challenges the agent and promotes learning of general navigation skills rather than fixed routes.

These techniques work together to create a robust learning environment for the TD3 agent, allowing it to effectively learn in a continuous action space despite the inherent challenges.

## YouTube Video of the inference simulation 
 