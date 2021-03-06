# DRL-Navigation

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

This project was completed using the workspace provided by Udacity.  

There is also an option to install/configure this environment locally: https://github.com/udacity/deep-reinforcement-learning#dependencies

This will also require downloading the Unity environment from Udacity.

### Instructions
Running the Navigation.ipynb notebook will train the agent.  The training will stop early when an average score of 13 (over a sliding window of 100 episodes) is achieved, and the model weights for the Deep Q Network (trained policy used to predict actions based on input state) will be saved in checkpoint.pth.
