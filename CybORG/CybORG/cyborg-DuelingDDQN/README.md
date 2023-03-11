# Cage-submission

Due to red agents’ behaviours not changing mid-episode, and the fact that they are predictable, we thought that fingerprinting the agent we are facing and then assigning it to a trained model made the most sense. If multiple red agents could exist in the environment in parallel, or if the red agents could change behaviour mid-episode, or if noise was added (Green Agent), then we would have applied hierarchical RL or utilised an RNN (which we expect to do in the second version of the challenge).
In addition, due to the action space being small (the blue agent cannot perform multiple actions at once, i.e restore multiple hosts for instance), we felt that reinforcement learning was appropriate, however in reality the action spaces for the defender (and attacker) would be too large for our approach.

As a result, we trained two models using DDQN for B_line and Meander. We also experimented with regular Q-learning for B_line after reducing the action and observation spaces, this was successful, but is not included in this submission as it does not add any value. This approach was however interesting to analyse the largest and smallest Q-values to confirm our suspicions.

Finally, it should be noted that we have not considered the Misinform action because it was not in the initial release. This made sense as the Green Agent does not figure in the evaluation.

# Agents

We built three agents:
1. A Sleep blue agent 
2. A DDQN blue agent
3. A Main blue agent which fingerprints the red agents and assigns a blue agent

The agents can be found in the Agents folder.

# DDQN

The DDQN implementation was taken from the following Github page https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/tree/master/DDQN where it has not been modified except for the model architecture (we opted for a MLP instead of CNN). The architecture is as follows for both models:

        self.fc1 = nn.Linear(input_dims[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

Where fc1 and fc2 have ReLU activations.

We trained two models: one for B_line and one for Meander. These are stored in the Models folder.

The train.py and utils.py files are included in the root directory for completeness but are not called in the evaluation.

# Evaluation

The Evaluation folder contains the evaluation.py file and an .md file discussing our approach's strengths and weaknesses.
# Dependencies

There are very few dependencies: pytorch (with cuda), numpy and matplotlib.

In addition, we have not update the challenge to version 1.2, therefore the action space for the blue agent remains Discrete(41).

# Thank you

We would like to thank the organisers of the challenge, and we look forward to version 2.
