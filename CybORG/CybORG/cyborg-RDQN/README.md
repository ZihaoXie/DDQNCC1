# Cage-submission

This approach rotated the red agents (Sleep, Meander and B_line) during training epochs of a recurrent neural network double DQN. 

RNN was chosen in order to give the model some memory of agent's past actions in order to help distinguish them - we found that RNNs with a memory of length 16 or 32 steps tended to outperform those with only 8 steps

We used a random search to find the optimal hyperaparameter configuration

Limitations of this approach include that it has only been exposed and tested on the same kinds of Red agents and we do not know how behaviour will generalise

This is our version 2 submission for cyborg-v1.2 (version 1 was for cyborg v.1.1)

# RNN-DDQN

The DDQN implementation was taken from the following Github page https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/tree/master/DDQN where it has not been modified except for the model architecture (we opted for a MLP instead of CNN). The architecture is as follows for both models:

but modified to use Gated Recurrent Unit (GRU) layers, allowing the model to 'remember' previous activity

The hyperaperameters chosen for the best-performing model were:

depth: 2 layers \
neurons in hidden layers: 64 \
number of previous steps to consider: 16 \
gamma (discount factor for future rewards): 0.5 \
epsilon (chance of picking a random action at start of training): 0.5 \
epsilon decrease rate: 5e-06 \
minimum epsilon: 0.1 \
learning rate: 0.0001 \
memory size: 5000 \
replace memory frequency (episodes): 500 \
length of episodes: 100 \
number of episodes: 1000 \
batch size: 32 

# Evaluation

Evaluation can be triggered by running evaluation/evaluation.py

# Agents

The code implementing the agent can be found in evaluation/MainAgent.py

# Wrapper

The wrapper used was the CyborgChallengeWrapper 

# Dependencies

Cyborg version 1.2 \
pandas==1.3.4 (for training only) \
numpy==1.21.4 \
torch==1.10.0

# Thank you

We would like to thank the organisers of the challenge, and we look forward to version 2.
