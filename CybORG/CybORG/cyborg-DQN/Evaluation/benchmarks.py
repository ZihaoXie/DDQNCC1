import numpy as np
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent, BlueReactRestoreAgent, BlueReactRemoveAgent, SleepAgent
import random
import inspect

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

# added to verify that our approach outperforms BlueReactRestoreAgent and BlueReactRemoveAgent
def benchmark_checks(eps_len=100, num_eps=10, restore=True, red_agent=RedMeanderAgent):
    agents = {
        'Red': red_agent}
    scores = []

    if restore:
        blue_agent = BlueReactRestoreAgent()
        blue_agent_name = "Restore"
    else:
        blue_agent = BlueReactRemoveAgent()
        blue_agent_name = "Remove"

    for i in range(num_eps):
        score = 0
        env = CybORG(PATH, 'sim', agents=agents)
        results = env.reset('Blue')
        obs = results.observation
        action_space = results.action_space
        for j in range(eps_len):
            action = blue_agent.get_action(obs, action_space)
            results = env.step(agent="Blue", action=action)
            obs = results.observation
            score += results.reward

        scores.append(score)

    avg_score = np.mean(scores)
    print('average score {} for {} on eps_len {} and num_eps {}, and defender {}'
          ''.format(avg_score, red_agent.__name__, eps_len, num_eps, blue_agent_name))

if __name__ == '__main__':
    random.seed(1)
    for num_eps in [10, 100]:
        for eps_len in [30, 50, 100]:
            for agent in [RedMeanderAgent, B_lineAgent]:
                for agent_def in [True, False]:
                    benchmark_checks(eps_len=eps_len, num_eps=num_eps, restore=agent_def, red_agent=agent)