# Copyright DST Group. Licensed under the MIT license.

# Cyber Operations Research Gym (CybORG)

A cyber security research environment for training and development of security human and autonomous agents. Contains a common interface for both emulated, using cloud based virtual machines, and simulated network environments.

## Installation

Install CybORG locally using pip

```
# from the cage-challenge-1/CybORG directory
pip install -e .
```
## Train The agent(DDQN/RNNDQN)

```
#DDQN agent 
# cd \CybORG\CybORG\cyborg-DQN

python train.py


# RNNDQN agent
There are some small bugs in DQNAgent.py in the source code, I made some modifications.

# cd \CybORG\CybORG\cyborg-RDQN

python train.py
```

