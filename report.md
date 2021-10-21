# DRL-Navigation

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Algorithm

For this project I implemented the Deep Q-Network as described in the paper: [Human-level control through deep reinforcement
learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

This approach moves beyond standard online Q-Learning in two respects:
* Experience replay
* Use a separate network for generating the targets in the Q-learning update

Both of these improvements help to address instabilities associated with Q-Learning.  The paper describes it as follows:

> We address these instabilities with a novel variant of Q-learning, which
> uses two key ideas. First, we used a biologically inspired mechanism
> termed experience replay that randomizes over the data, thereby
> removing correlations in the observation sequence and smoothing over
> changes in the data distribution. Second, we used
> an iterative update that adjusts the action-values (Q) towards target
> values that are only periodically updated, thereby reducing correlations
> with the target.

The pseudo-code from the paper is as follows:

![alg](screencapture.png)

### Implementation

The code used for this project is heavily based on the solution in the Deep Q-Learning Lunar Lander project.  The key components are as follows:

#### model.py:  
This file contains the QNetwork class, which implements a pytorch deep neural network that takes input the size of the state space, produces output the size of the action space, and has two hidden layers of configurable size.  For this project, both layers were set to 512 nodes.  As described in the algorithm section, the QNetwork(s) were used for both training and inference at various points.

#### dqn_agent.py:
This file contains two classes.  The Agent class is the main interface for interacting with and learning from the enviroment.  The Agent delegates to the ReplayBuffer class to store the experience tuples needed for experience replay.  The key methods are described below.
##### Agent.__init__
A key feature of this constructor is that it instantiates two QNetworks, qnetwork_local and qnetwork_target.  While qnetwork_local is being trained, qnetwork_target is used to generate stable targets for computing the loss.
##### Agent.act
This method returns an action based on the input state.  It makes an epsilon greedy selection based on the prediction from qnetwork_local.  Since epsilon is an input parameter, the training loop can gradually move from exploration to exploitation.
##### Agent.step
The training loop will choose an action and provide it to the enviroment.  With the environment's response, we have a full experience (state, action, reward, next_state, done) that can be stored in the ReplayBuffer.  For every UPDATE_EVERY calls to Agent.step, the method will sample BATCH_SIZE samples from the ReplayBuffer and update both QNetworks by calling Agent.learn
##### Agent.learn
This method implements the target computation and gradient descent step from the algorithm pseudo-code, with the modification of using an Adam optimizer.  The updated qnetwork_local is then used to gradually update (based on interpolation parameter TAU) the qnetwork_target.
#### Network.ipynb
blah




The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

Plot of Rewards

Ideas for Future Work
