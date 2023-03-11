# Copyright DST Group. Licensed under the MIT license.

# Cyber Operations Research Gym (CybORG)

A cyber security research environment for training and development of security human and autonomous agents. Contains a common interface for both emulated, using cloud based virtual machines, and simulated network environments.

## Installation

Install CybORG locally using pip

```
# from the cage-challenge-1/CybORG directory
pip install -e .
```
## Train DDQN Agent

```
#DDQN Agent

# cd \CybORG\CybORG\cyborg-DQN

python train.py
```

## Train RNNDQN Agent
```
There are some bugs in DQNAgent.py in the source code, I made some modifications.

#RNNDDQN Agent

# cd \CybORG\CybORG\cyborg-RDQN

python train.py
```
## DuelingDDQN Agent
```
cd \CybORG\CybORG\cyborg-DuelingDDQN
python train.py
```


## References:
```
https://github.com/john-cardiff/cyborg-cage
https://github.com/mprhode/cyborg-submission-CUABv2
```

