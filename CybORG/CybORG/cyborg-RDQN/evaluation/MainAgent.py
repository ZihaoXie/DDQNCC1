from DQN.DQNAgent import RNNDQNAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent

class MainAgent(BaseAgent):
    def __init__(self, suffix=31):
        self.end_episode()

        self.agent = RNNDQNAgent(
            input_dims=(52,),
            n_actions=54,
            lookback_steps=16,
            epsilon=0, chkpt_dir="../saved_best_model",
            algo=f'RNNDDQNAgent_{suffix}',
            env_name='Scenario1b')
        self.agent.load_models()

    def get_action(self, observation, action_space=None):
        return self.agent.get_action(observation, action_space=action_space)

    def train(self):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass
