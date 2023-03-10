from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from Agents.DQNAgent import DQNAgent
from Agents.BlueSleepAgent import BlueSleepAgent


class MainAgent(BaseAgent):
    def __init__(self):
        self.end_episode()
        # wake_up() and wake_up_value is to handle a rare case
       # 它会记住它分配的前一个代理，以防红色代理回到睡眠状态并认为它面对的是睡眠智能体
       # it remembers the previous agent it assigned in case the red agent goes back to sleep thinking it is facing a Sleep agent
        self.wake_up_value = 0
        self.previous_agent = "Sleep"

    def get_action(self, observation, action_space=None):
        self.wake_up_value = self.wake_up_value - 1 # 通过减少属性的值，main agent可以保留之前分配给他的agent信息
        # 通过现在的obs 和上一时刻的obs连起来组成新的列表，并且用来确定该使用哪个智能体进行下一步的行动
        previous_two_observations = list(observation + self.last_observation)
        # pick agent based on a fingerprint  根据fingerprint 选择哪个智能体
        self.assign_agent(previous_two_observations) # 指派agent
        self.last_observation = observation # 将当前的obs 赋值给last_obs 以便在下一次调用的时候能用
        #  睡眠且唤醒值 大于97 而且 obs 和大于0 调用唤醒
        if self.agent_name == "Sleep" and self.wake_up_value > 97 and sum(observation) > 0:
            self.wake_up()
        return self.agent.get_action(observation)


    #  分配智能体
    # 根据之前合并的观察数组来确定 使用哪个训练好的agent模型 来应对 red agent
    def assign_agent(self, previous_two_observations):

        # fingerprints are sum of both previous observation bits
        # fingerprints是两个先前观察位的总和
        # 定义了fingerprints 的种类
        sleep_fingerprinted = [0] * 52
        meander_fingerprinted = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bline_fingerprinted = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bline_fingerprinted_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # 如果之前的观察符合其中的一种，则 用那种agent来应对

        if previous_two_observations == meander_fingerprinted:
            self.use_meander()

        elif previous_two_observations == bline_fingerprinted or previous_two_observations == bline_fingerprinted_2:
            self.use_bline()

        # stick with sleep
        elif previous_two_observations == sleep_fingerprinted:
            # entering a sleep through fingerprint remembers the previous agent it assigned
            # 通过指纹进入睡眠会记住它分配的先前代理
            self.wake_up_value = 100
            self.previous_agent = self.agent_name
            self.end_episode()
        else:
            pass

    def wake_up(self):  # 唤醒功能 发现红色智能体并不是sleep agent
        if self.previous_agent == "Meander":
            self.use_meander()
            self.wake_up_value = 0
        elif self.previous_agent == "Bline":
            self.use_bline()
            self.wake_up_value = 0
        else:
            pass

    def use_meander(self):  # 使用已经训练好的模型 已应对红色meander agent
        self.agent = DQNAgent(chkpt_dir="../Models/model_meander/", algo='DDQNAgent', env_name='Scenario1b')
        # needed to get the pytorch checkpoint
        self.agent.load_models()
        self.agent_name = "Meander"

    def use_bline(self):  #  使用已经训练好的DDQN 模型应对 红色bline agent
        self.agent = DQNAgent(chkpt_dir="../Models/model_b_line/", algo='DDQNAgent', env_name='Scenario1b')
        # needed to get the pytorch checkpoint
        self.agent.load_models()
        self.agent_name = "Bline"

    def train(self):
        pass

    def end_episode(self):
        self.last_observation = [0] * 52
        # we start with the sleep agent, we might want to build another one though as the default
        self.agent = BlueSleepAgent()
        self.agent_name = "Sleep"

    def set_initial_values(self, action_space, observation):
        pass
