# Car driving simulation using TD3

This repo demonstrates how to train a reinforcement learning model using Twin-Delayed Deep Deterministic (TD3) policy gradient. A city map and kivy package is used to create the environment for simulation.

## New environment
![alt text](images_new/citymap.png)  
 
The above city map is used for training the simulation. Following mask is created to map the roads.  
![alt text](images_new/MASK1.png)  
 
There are 3 destinations on the map which are randomly selected in each turn and once selected shown with following icon on the map.  
![alt text](images_new/target.png)  

## TD3 model training tricks
One of primary difference of in this model is that the action space in continuous. i.e. instead of the model deciding on one of available options as next action, model will predict a value that will be directly used as action. In this case, the model predicts are angle value that is directly used to rotate the car (instead of selecting an angle from a range of available options). Therefore the training of this model comparatively difficult. The TD3 molde training involved certain tricks and configurations. Some of them are described below.
 1. The replay memory is set to 20k steps so that we have enough examples of positive rewards (when car is moving towards the target and on the reoad) before it starts training.
 2. The car is initialised at different random places after every 1000 steps in that first 20k steps. These random points are selected after diving the map in 12 blocks.

## Youtube Video of the training simulation
